"""Learn future threat scores in the current camera frame from panorama history."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
import time
import struct
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def image_history(rgb, depth, valid):
    """RGB uint8, radial depth in metres, with leading time and trailing RGB axes."""
    color = torch.as_tensor(rgb).permute(0, 3, 1, 2).float() / 255.
    d = torch.as_tensor(depth)[:, None]
    ok = torch.isfinite(d) & (d > 0)
    channels = torch.cat((color, torch.where(ok, d.clamp(max=50)/50., 0.), ok.float()), dim=1)
    return channels * torch.as_tensor(valid)[:, None, None, None]


class RGBDField(nn.Module):
    def __init__(self, wide=False):
        super().__init__()
        channels = (5,24,40,64) if wide else (5,16,32,48)
        hidden = 96 if wide else 64
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i, o, 3, stride=2), nn.GroupNorm(4, o), nn.SiLU())
            for i, o in zip(channels,channels[1:])])
        self.pool = nn.AdaptiveAvgPool2d((2, 8))
        self.temporal = nn.GRU(channels[-1]*2*8, 64, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(68, 96), nn.SiLU(), nn.Linear(96, hidden),
                                     nn.SiLU(), nn.Linear(hidden, 1))

    def feature_history(self, images):
        batch, frames = images.shape[:2]
        h = images.flatten(0,1)
        for block in self.blocks:
            h = F.pad(h, (1,1,0,0), mode="circular")
            h = block(F.pad(h, (0,0,1,1), mode="replicate"))
        return h.reshape(batch,frames,*h.shape[1:])

    def temporal_context(self, maps):
        batch,frames = maps.shape[:2]
        _,last = self.temporal(self.pool(maps.flatten(0,1)).reshape(batch,frames,-1))
        return last[0]

    def encode(self, images):
        return self.temporal_context(self.feature_history(images))

    def query_logits(self, latent, points_frd, tau):
        context = latent[:, None].expand(-1, points_frd.shape[1], -1)
        return self.decoder(torch.cat([context, points_frd/3., tau/1.6], dim=-1)).squeeze(-1)

    def forward(self, images, points_frd, tau):
        return self.query_logits(self.encode(images), points_frd, tau).sigmoid()


class ProjectedRGBDField(RGBDField):
    """Query the latest ERP in FRD, alongside the original four-frame context."""
    def __init__(self):
        super().__init__()
        self.decoder[0] = nn.Linear(68+48+5+2,96)
        steps=torch.arange(-2,3)
        dy,dx=torch.meshgrid(steps,steps,indexing="ij")
        self.register_buffer("offsets", torch.stack([dx.flatten(),dy.flatten()]), persistent=False)

    def encode(self, images):
        maps=self.feature_history(images)
        return self.temporal_context(maps), maps[:,-1], images[:,-1]

    def local_query(self, feature_map, image, points):
        batch,channels,height,width=image.shape
        x,y,z=points.unbind(-1)
        horizontal=torch.linalg.vector_norm(points[...,:2],dim=-1)
        # At vertical rays longitude is arbitrary; choose the front meridian.
        lon=torch.atan2(y,torch.where(horizontal>1e-8,x,torch.ones_like(x)))
        lat=torch.atan2(z,horizontal.clamp_min(1e-8))
        u=(lon/(2*torch.pi)+.5)*width
        v=(lat/torch.pi+.5)*height
        ix=(u.floor().long()[...,None]+self.offsets[0])%width
        iy=(v.floor().long()[...,None].clamp(0,height-1)+self.offsets[1]).clamp(0,height-1)
        indices=(iy*width+ix).reshape(batch,1,-1).expand(-1,channels,-1)
        patch=image.flatten(2).gather(2,indices).reshape(batch,channels,points.shape[1],-1)
        direct=patch[:,:,:,12].transpose(1,2)
        lon_px=((ix+.5)/width-.5)*(2*torch.pi)
        lat_px=((iy+.5)/height-.5)*torch.pi
        rays=torch.stack([lat_px.cos()*lon_px.cos(),lat_px.cos()*lon_px.sin(),lat_px.sin()],dim=-1)
        surface=rays*(50*patch[:,3])[...,None]
        clearance=torch.linalg.vector_norm(surface-points[...,None,:],dim=-1)
        clearance=clearance.masked_fill(patch[:,4]<.5,8.).amin(dim=-1).clamp(max=8.)
        residual=(50*direct[...,3]-torch.linalg.vector_norm(points,dim=-1)).clamp(-8.,8.)*direct[...,4]

        # CNN stride is 8; the receptive field of cell j is centered on pixel 8*j.
        fh,fw=feature_map.shape[-2:]
        fx=((u-.5)/8)%fw
        fy=((v-.5)/8).clamp(0,fh-1)
        grid=torch.stack([2*(fx+1.5)/(fw+2)-1, 2*(fy+.5)/fh-1],dim=-1)
        wrapped=F.pad(feature_map,(1,1,0,0),mode="circular")
        local=F.grid_sample(wrapped,grid[:,:,None],align_corners=False,padding_mode="border")[:,:,:,0]
        return torch.cat([local.transpose(1,2),direct,residual[...,None],clearance[...,None]],dim=-1)

    def query_logits(self, encoded, points_frd, tau):
        context,maps,image=encoded
        local=self.local_query(maps,image,points_frd)
        shared=context[:,None].expand(-1,points_frd.shape[1],-1)
        return self.decoder(torch.cat([shared,points_frd/3.,tau/1.6,local],dim=-1)).squeeze(-1)


class GateFrames:
    def __init__(self, boxes):
        self.centers = np.array([b["center"] for b in boxes]).reshape(-1,3)
        self.half = np.array([b["size"] for b in boxes]).reshape(-1,3)/2.
        self.rot = Rotation.from_euler("z", [[b["yaw"]] for b in boxes]).as_matrix() if boxes else None

    def distances(self, points):
        if not len(self.centers):
            return np.full(len(points),np.inf)
        local = np.einsum("qbi,bij->qbj", points[:,None]-self.centers, self.rot)
        delta = np.abs(local)-self.half
        sdf = np.linalg.norm(np.maximum(delta,0.), axis=-1) + np.minimum(delta.max(axis=-1),0.)
        return sdf.min(axis=1)

    def sample(self, rng, anchor, count):
        near = np.argsort(np.linalg.norm(self.centers-anchor,axis=1))[:8]
        idx = rng.choice(near,count)
        local = rng.uniform(-1.,1.,(count,3))*self.half[idx]
        noise = rng.normal(size=(count,3))*rng.uniform(0.,.65,(count,1))
        return self.centers[idx]+np.einsum("bij,bj->bi", self.rot[idx], local)+noise


def threat_labels(query, future, gates, gate_margin):
    neighbour = (np.linalg.norm(query-future, axis=1)-.55)/.08
    frame = (gates.distances(query)-gate_margin)/.05
    return 1/(1+np.exp(np.clip(np.minimum(neighbour,frame),-60,60)))


def half_size_observations(rgb, depth):
    """Match the online 2x area RGB / nearest-exact range resize before window sampling."""
    if rgb.shape[-3:-1] != (256,512):
        raise ValueError("Expected 512x256 source panoramas")
    color = np.empty((*rgb.shape[:2],128,256,3),np.float16)
    for start in range(0,len(rgb),8):
        block = rgb[start:start+8]
        sums = block[:, :, 0::2, 0::2].astype(np.uint16)
        sums += block[:, :, 0::2, 1::2]
        sums += block[:, :, 1::2, 0::2]
        sums += block[:, :, 1::2, 1::2]
        color[start:start+8] = sums.astype(np.float32)*.25
    return color, depth[:,:,1::2,1::2].copy()


def mapped_observation(path, name):
    """Map a contiguous NPY member of an uncompressed NPZ without loading the full flight."""
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(name+".npy")
    if info.compress_type!=zipfile.ZIP_STORED or info.flag_bits&1:
        raise ValueError(f"Expected an uncompressed observation in {path}")
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        header=stream.read(30)
        if header[:4]!=b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        filename_len,extra_len=struct.unpack_from("<HH",header,26)
        stream.seek(filename_len+extra_len,1)
        start=stream.tell()
        version=np.lib.format.read_magic(stream)
        if version!=(1,0):
            raise ValueError("Expected NPY version 1.0")
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        offset=stream.tell()
        if fortran or dtype.hasobject or offset-start+int(np.prod(shape))*dtype.itemsize!=info.file_size:
            raise ValueError("Invalid packed observation")
    return np.memmap(path,dtype=dtype,shape=shape,offset=offset,mode="r")


class GateQueries:
    """Training-only queries on the gate plane, labelled from the recorded flight."""
    def __init__(self, episode, margin):
        times, states, _, _, _, gates, eligible = episode
        bars = gates.centers.reshape(-1, 4, 3)
        self.centers = bars.mean(axis=1)
        side = bars[:, 1] - bars[:, 0]
        self.side = side / np.linalg.norm(side, axis=1, keepdims=True)
        self.pairs = []
        self.empty = []
        for ego in (0, 1):
            peer = states[:, 1-ego, :3]
            now = peer[eligible]
            future = np.column_stack([np.interp(times[eligible]+.4, times, axis) for axis in peer.T])
            observer = states[eligible, ego, :3]
            for gate, center in enumerate(self.centers):
                # The 4 cm jitter used below stays within these distance margins.
                offset = future-center
                query = center + np.clip(offset @ self.side[gate], -.35, .35)[:, None]*self.side[gate]
                query[:, 2] += np.clip(offset[:, 2], -.35, .35)
                near = np.linalg.norm(observer-center, axis=1) <= 7
                reached = (np.linalg.norm(future-query, axis=1)<.32) & (np.linalg.norm(now-query, axis=1)>.85)
                reached &= gates.distances(query)>margin+.15
                for k in np.flatnonzero(near & reached):
                    self.pairs.append((eligible[k], ego, gate, query[k]))
                clear = (np.linalg.norm(now-center, axis=1)>1.35) & (np.linalg.norm(future-center, axis=1)>1.35)
                if gates.distances(center[None])[0] > margin+.30:
                    self.empty.extend((eligible[k], ego, gate, center) for k in np.flatnonzero(near & clear))
        if not self.pairs or not self.empty:
            raise ValueError("Training flight lacks clear or arrival gate queries")

    def sample(self, rng, arrival):
        pool = self.pairs if arrival else self.empty
        frame, ego, gate, center = pool[rng.integers(len(pool))]
        radius = .04 if arrival else .12
        offsets = rng.uniform(-radius, radius, (6, 2))
        offsets[0] = 0
        points = center + offsets[:, :1]*self.side[gate]
        points[:, 2] += offsets[:, 1]
        return frame, ego, np.repeat(points, 2, axis=0), np.tile([0., .4], 6)


class Windows(Dataset):
    def __init__(self, manifest, split, samples, gate_margin, half_size=False, lazy_windows=False, gate_pairs=False):
        self.episodes = []
        self.resize_window = lazy_windows and half_size
        self.samples, self.gate_margin = samples, gate_margin
        self.seed = 6106 if split=="train" else 19
        self.epoch = 0
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        seen = set()
        for e in data["episodes"]:
            source = (manifest.parent/e["trajectory"]).resolve()
            key = e.get("trajectory_sha256",str(source))
            if key in seen or str(source) in seen:
                raise ValueError("Duplicate trajectory in manifest")
            seen.update([key,str(source)])
            if e["split"] != split:
                continue
            with np.load(source,allow_pickle=False) as f:
                times, states = f["time"], f["state"]
            path=manifest.parent/e["observations"]
            with np.load(path,allow_pickle=False) as f:
                valid=f["valid"]
                if lazy_windows:
                    rgb,depth=(mapped_observation(path,key) for key in ("rgb","depth"))
                else:
                    rgb,depth=f["rgb"],f["depth"]
                if f["time"].shape != times.shape or not np.allclose(times,f["time"],rtol=0,atol=1e-4):
                    raise ValueError("RGB-D timestamps differ from trajectory")
            if (np.any(np.diff(times)<=0) or states.shape[:2] != (len(times),2) or
                    rgb.shape[:2] != states.shape[:2] or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8
                    or depth.shape!=rgb.shape[:-1] or valid.shape!=states.shape[:2]):
                raise ValueError("Invalid episode dimensions or timestamps")
            if half_size or lazy_windows:
                if not valid.all() or not np.allclose(np.diff(times),.02,atol=1e-5,rtol=0):
                    raise ValueError("Expected complete 50 Hz RGB-D recordings")
                if half_size and not lazy_windows:
                    rgb, depth = half_size_observations(rgb,depth)
            eligible = np.flatnonzero((np.arange(len(times))>=3) & (times<=times[-1]-1.6))
            if not len(eligible):
                raise ValueError("Episode requires four history frames and 1.6 s of future labels")
            self.episodes.append((times, states, rgb, depth, valid, GateFrames(e["gate_boxes"]), eligible))
        if not self.episodes:
            raise ValueError(f"No {split} episodes")
        self.gate_queries = [GateQueries(ep, gate_margin) for ep in self.episodes] if gate_pairs else []

    def __len__(self):
        return self.samples

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + self.epoch*self.samples + i)
        times, states, rgb, depth, valid, gates, eligible = self.episodes[i%len(self.episodes)]
        frame, ego = rng.choice(eligible), rng.integers(2)
        # Retain the original distribution of observation times in half the windows.
        group = (i // len(self.episodes)) % 4
        targeted = bool(self.gate_queries) and group >= 2
        if targeted:
            arrival = group == 3
            sampler = self.gate_queries[i % len(self.episodes)]
            frame, ego, extra_q, extra_tau = sampler.sample(rng, arrival)
        h = slice(frame-3,frame+1)
        if self.resize_window:
            color,ranges=half_size_observations(rgb[h,ego:ego+1],depth[h,ego:ego+1])
            images=image_history(color[:,0],ranges[:,0],valid[h,ego])
        else:
            images=image_history(rgb[h,ego], depth[h,ego], valid[h,ego])
        anchor = states[frame,ego,:3]
        tau = rng.integers(0,81,32)*.02
        future = np.column_stack([np.interp(times[frame]+tau,times,states[:,1-ego,k]) for k in range(3)])
        direction = rng.normal(size=(32,3))
        direction /= np.linalg.norm(direction,axis=1,keepdims=True)
        query = future + direction*rng.uniform(0.,1.25,(32,1))
        query[-8:] = anchor + rng.uniform(-6.,6.,(8,3))
        if len(gates.centers):
            query[12:24] = gates.sample(rng,anchor,12)
        if targeted:
            extra_future = np.column_stack([np.interp(times[frame]+extra_tau,times,axis)
                                            for axis in states[:,1-ego,:3].T])
            query = np.vstack((query, extra_q))
            tau = np.r_[tau, extra_tau]
            future = np.vstack((future, extra_future))
        label = threat_labels(query, future, gates, self.gate_margin)
        if targeted:
            paired = label[-12:].reshape(-1, 2)
            if np.any(paired[:,0] >= .1) or (np.any(paired[:,1] <= .9) if arrival else np.any(paired[:,1] >= .1)):
                raise ValueError("Unexpected label for targeted gate query")
        elif self.gate_queries:
            # Equal batch shapes with an unbiased sample of the original queries.
            extra = rng.choice(len(query), 12, replace=False)
            query, tau, label = np.vstack((query,query[extra])), np.r_[tau,tau[extra]], np.r_[label,label[extra]]
        # Planner wxyz is body FLU -> world. ERP camera axes are forward/right/down.
        rotation = Rotation.from_quat(states[frame,ego,[7,8,9,6]])
        camera_points = rotation.inv().apply(query-anchor)*[1,-1,-1]
        return (images, *[torch.as_tensor(a,dtype=torch.float32) for a in (camera_points,tau[:,None],label)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--total-epochs",type=int,help="Stop at this total epoch count when resuming")
    ap.add_argument("--batch-rest-ms",type=float,default=0.,help="Idle after each completed batch to reduce sustained load")
    ap.add_argument("--lazy-windows",action="store_true",help="Load only a single four-frame history per sample")
    ap.add_argument("--samples",type=int,default=4096)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--gate-margin",type=float,default=.25,help="Gate surface clearance in metres at score 0.5")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--resume",type=Path)
    ap.add_argument("--projected",action="store_true",help="Add query-aligned RGB-D and local CNN features")
    ap.add_argument("--wide", action="store_true", help="Wider native-resolution global CNN and query decoder")
    ap.add_argument("--half-size",action="store_true",help="Match online 512x256 to 256x128 preprocessing")
    ap.add_argument("--gate-pairs", action="store_true", help="Oversample gate arrivals and empty gate centers in training")
    args=ap.parse_args()
    if args.wide and (args.half_size or args.projected):
        ap.error("--wide uses the native-resolution global encoder")
    if min(args.epochs,args.samples,args.batch_size)<=0 or args.gate_margin<0:
        ap.error("Positive counts and nonnegative gate margin required")
    if args.batch_rest_ms<0 or (args.total_epochs is not None and not args.resume):
        ap.error("A total epoch target requires a checkpoint; batch rest must be nonnegative")
    torch.manual_seed(6106)
    torch.set_num_threads(1)
    train=Windows(args.manifest,"train",args.samples,args.gate_margin,args.half_size,args.lazy_windows,args.gate_pairs)
    val=Windows(args.manifest,"validation",min(512,args.samples),args.gate_margin,args.half_size,args.lazy_windows)
    manifest_info=json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    model=(ProjectedRGBDField() if args.projected else RGBDField(wide=args.wide)).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=.0015)
    first,best=0,float("inf")
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=True)
        if ck["gate_margin"]!=args.gate_margin:
            raise ValueError("Gate margin differs from checkpoint")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        first,best=ck["epoch"]+1,ck["best"]
    stop=args.total_epochs if args.total_epochs is not None else first+args.epochs
    if stop<=first:
        ap.error("Checkpoint already reached the total epoch target")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"config.json").write_text(json.dumps(vars(args),default=str,indent=2))
    if args.resume:
        original_best=args.resume.with_name("best.pt")
        if not original_best.exists():
            raise FileNotFoundError(original_best)
        from shutil import copyfile
        copyfile(original_best,args.output/"best.pt")
    for epoch in range(first,stop):
        train.epoch=epoch
        row={"epoch":epoch}
        for name,ds in [("train",train),("validation",val)]:
            model.train(name=="train")
            total=0.
            with torch.set_grad_enabled(model.training):
                for batch in DataLoader(ds,batch_size=args.batch_size):
                    images,q,tau,y=(v.to(args.device) for v in batch)
                    logits=model.query_logits(model.encode(images),q,tau)
                    loss=F.binary_cross_entropy_with_logits(logits,y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite risk BCE")
                    if model.training:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        opt.step()
                    total+=loss.item()*len(images)
                    if args.batch_rest_ms:
                        time.sleep(args.batch_rest_ms/1000.)
            row[name+"_bce"]=total/len(ds)
        improved=row["validation_bce"]<best
        best=min(best,row["validation_bce"])
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),best=best,epoch=epoch,
                gate_margin=args.gate_margin,query_frame="current_camera_FRD",inputs="four_RGBD_frames",
                projected=args.projected,wide=args.wide,training_manifest=str(args.manifest.resolve()),
                vehicle_geometry=manifest_info.get("vehicle_geometry"),half_size=args.half_size)
        for dest in (["best","last"] if improved else ["last"]):
            tmp=args.output/(dest+".tmp")
            torch.save(ck,tmp)
            tmp.replace(args.output/(dest+".pt"))
        with (args.output/"history.jsonl").open("a") as f:
            print(json.dumps(row),file=f)
        print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
