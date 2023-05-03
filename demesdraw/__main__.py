import argparse

import demes
import matplotlib.pyplot as plt

import demesdraw


class Command:
    def __init__(self, subparsers, subcommand):
        self.parser = subparsers.add_parser(
            subcommand,
            help=self.__doc__,
            description=self.__doc__,
        )
        self.parser.set_defaults(func=self)
        self.parser.add_argument(
            "--log-time",
            action="store_true",
            default=False,
            help="Use a log-10 scale for the time axis.",
        )
        self.parser.add_argument(
            "--aspect",
            type=float,
            help="Set the aspect ratio (height/width) of the plot.",
        )
        self.parser.add_argument(
            "--scale",
            type=float,
            help="Scale the figure size by the given value.",
        )
        self.parser.add_argument(
            "--title", type=str, help="Set the title of the figure to the given string."
        )
        self.parser.add_argument(
            "input_file",
            metavar="model.yaml",
            type=argparse.FileType(),
            help=(
                "Filename of the model. The special value '-' may be used to "
                "read from stdin. The file may be in any format supported by "
                "the demes library."
            ),
        )
        self.parser.add_argument(
            "output_file",
            metavar="figure.img",
            type=str,
            default=None,
            help=(
                "The filename for the image. The file extension determines "
                "the filetype, and can be any format supported by Matplotlib "
                "(e.g. pdf, svg, png). If no file is specified, "
                "an interactive plot window will be opened."
            ),
            nargs="?",
        )


class TubesCommand(Command):
    """
    Plot a demes-as-tubes schematic of the model and the demes' relationships.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "tubes")

        self.parser.add_argument(
            "--scale-bar",
            action="store_true",
            default=False,
            help="Draw a scale bar that indicates population size.",
        )

    def __call__(self, args):
        graph = demes.load(args.input_file)
        fig, ax = demesdraw.utils.get_fig_axes(aspect=args.aspect, scale=args.scale)
        demesdraw.tubes(
            graph,
            ax=ax,
            log_time=args.log_time,
            title=args.title,
            scale_bar=args.scale_bar,
        )
        if args.output_file is not None:
            fig.savefig(args.output_file)
        else:
            # interactive plot
            plt.show()


class SizeHistoryCommand(Command):
    """
    Plot population size as a function of time for each deme in the model.
    """

    def __init__(self, subparsers):
        super().__init__(subparsers, "size_history")

    def __call__(self, args):
        graph = demes.load(args.input_file)
        fig, ax = demesdraw.utils.get_fig_axes(aspect=args.aspect, scale=args.scale)
        demesdraw.size_history(graph, ax=ax, log_time=args.log_time, title=args.title)
        if args.output_file is not None:
            fig.savefig(args.output_file)
        else:
            # interactive plot
            plt.show()


def cli():
    top_parser = argparse.ArgumentParser(
        prog="demesdraw", description="Draw a Demes model."
    )
    top_parser.add_argument(
        "--version", action="version", version=demesdraw.__version__
    )
    subparsers = top_parser.add_subparsers(dest="subcommand")
    TubesCommand(subparsers)
    SizeHistoryCommand(subparsers)
    args = top_parser.parse_args()
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)
    args.func(args)


if __name__ == "__main__":
    cli()
