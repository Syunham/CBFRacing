"""Supervised five-event timing policies from saved rolling-flight labels."""
import argparse
import copy
import hashlib
from pathlib import Path

from cbfracing.core.context import HERE, np, torch, save_json
from cbfracing.networks.event_supervised import EventNetwork, ShortSegmentEventNetwork, load_event


def build_parser(parents):
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument('--name', required=True)
    parser.add_argument('--frontend', choices=['minimum-snap'], default='minimum-snap')
    parser.add_argument('--dataset', type=Path, default=HERE/'data/events_rollout.npz')
    parser.add_argument('--architecture', choices=['direct', 'short'], default='direct')
    parser.add_argument('--initial', type=Path, help='Frozen bounded policy for --architecture short')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--log-bound', type=float, default=.3)
    return parser


def main(args):
    if args.steps < 1 or args.learning_rate <= 0 or args.log_bound <= 0:
        raise ValueError('Steps, learning rate and residual bound must be positive.')
    with np.load(args.dataset, allow_pickle=False) as data:
        features = np.array(data['features'], dtype=np.float64)
        targets = np.array(data['targets'], dtype=np.float64)
        train = np.asarray(data['train'], dtype=int)
        valid = np.asarray(data['val'], dtype=int)
    if features.ndim != 2 or features.shape[1] != 54 or targets.shape != (len(features), 5):
        raise ValueError('Expected N x 54 features and N x 5 positive segment durations.')
    if not np.isfinite(features).all() or not np.isfinite(targets).all() or (targets <= 0).any():
        raise ValueError('Training data must be finite with strictly positive durations.')
    if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
        raise ValueError('Provide nonempty, disjoint training and validation indices.')
    if min(train.min(), valid.min()) < 0 or max(train.max(), valid.max()) >= len(features):
        raise ValueError('Dataset split indices are out of bounds.')
    torch.manual_seed(args.seed)
    x, y = torch.from_numpy(features), torch.from_numpy(targets)
    if args.architecture == 'short':
        initial = args.initial or HERE/'models/snap_bounded_rollout_v1/events.pt'
        base, metadata = load_event(initial)
        if metadata.get('architecture') != 'bounded_residual_v1':
            raise ValueError('Short residual training requires a bounded_residual_v1 checkpoint.')
        net = ShortSegmentEventNetwork(54, args.log_bound).double()
        net.base.load_state_dict(base.state_dict())
        mask = net.short_mask(x)
        if not mask[train].sum() or not mask[valid].sum():
            raise ValueError('Both splits require short-segment examples.')
        with torch.no_grad():
            prior = net.base(x).log()
        architecture = 'short_segment_residual_v1'
    else:
        if args.initial:
            raise ValueError('--initial is used by the short residual architecture.')
        net = EventNetwork(54, duration_min=.15).double()
        with torch.no_grad():
            net.xmean.copy_(x[train].mean(0))
            net.xstd.copy_(x[train].std(0, unbiased=False).clamp_min(1e-6))
            net.ymean.copy_(y[train].log().mean(0))
            net.ystd.copy_(y[train].log().std(0, unbiased=False).clamp_min(1e-6))
        mask = torch.ones_like(y)
        prior = None
        architecture = 'direct_supervised_v1'
    output = HERE/'models'/args.name
    output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.learning_rate)
    history, best, best_score = [], None, float('inf')
    for step in range(args.steps + 1):
        if step:
            prediction = net.log_times(x[train])
            loss = ((prediction-y[train].log()).square()*mask[train]).sum()/mask[train].sum()
            if prior is not None:
                loss = loss + .02*((prediction-prior[train]).square()*mask[train]).sum()/mask[train].sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
        if step % 25 == 0 or step == args.steps:
            with torch.no_grad():
                prediction = net(x[valid])
                score = float(((prediction.log()-y[valid].log()).square()*mask[valid]).sum()/mask[valid].sum())
                mae = float(((prediction-y[valid]).abs()*mask[valid]).sum()/mask[valid].sum())
            history.append(dict(step=step, validation_log_mse=score, validation_mae_s=mae))
            if score < best_score:
                best, best_score, best_step = copy.deepcopy(net.state_dict()), score, step
    checkpoint = dict(architecture=architecture, input_dim=54, duration_min=.15,
                      log_bound=args.log_bound, state=best, best_step=best_step)
    torch.save(checkpoint, output/'events.pt')
    net.load_state_dict(best)
    loaded, _ = load_event(output/'events.pt')
    with torch.no_grad():
        np.testing.assert_allclose(net(x).numpy(), loaded(x).numpy(), atol=1e-12)
    result = dict(architecture=architecture, steps=args.steps, best_step=best_step,
                  training_samples=len(train), validation_samples=len(valid), history=history,
                  dataset=str(args.dataset.resolve()), dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                  initial=str(initial.resolve()) if args.architecture == 'short' else None,
                  supervision='Executed subsequent gate intervals from audited rolling flights',
                  validation_scope='Held-out examples. Closed-loop flight evaluated separately.')
    save_json(output/'training.json', result)
    print('Saved', output/'events.pt', 'validation MAE seconds', min(history, key=lambda r:r['validation_log_mse'])['validation_mae_s'])
