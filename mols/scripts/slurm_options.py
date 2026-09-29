"""Shared, cluster-independent Slurm resource options."""


def add_slurm_options(parser):
    parser.add_argument("--partition", "--cluster", dest="partition", default=None,
                        help="Slurm partition (uses the scheduler default if omitted)")
    parser.add_argument("--account", default=None, help="Slurm account")
    parser.add_argument("--cpus-per-task", type=int, default=6)
    parser.add_argument("--mem", default="120G", help="Memory per node")
    parser.add_argument("--exclude", default=None, help="Slurm node list to exclude")


def slurm_cli(args):
    cmd = ["--cpus-per-task", str(args.cpus_per_task), "--mem", args.mem]
    for option in ("partition", "account", "exclude"):
        value = getattr(args, option)
        if value:
            cmd.extend([f"--{option}", value])
    return cmd
