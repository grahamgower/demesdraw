import pathlib
import subprocess
import tempfile

import pytest

import demesdraw
import tests


class TestToplevel:
    def test_help(self):
        out1 = subprocess.run(
            "python -m demesdraw -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m demesdraw --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    def test_no_arguments_produces_help_output(self):
        # If no params are given, the output is the same as --help.
        # But the returncode should be non-zero.
        out1 = subprocess.run("python -m demesdraw -h".split(), stdout=subprocess.PIPE)
        out2 = subprocess.run("python -m demesdraw".split(), stdout=subprocess.PIPE)
        assert out1.stdout == out2.stdout
        assert out2.returncode != 0

    def test_version(self):
        out = subprocess.run(
            "python -m demesdraw --version".split(),
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        assert out.stdout.strip() == demesdraw.__version__


class TestSubCommand:
    @pytest.mark.parametrize("subcommand", ["tubes", "size_history"])
    def test_help(self, subcommand):
        out1 = subprocess.run(
            f"python -m demesdraw {subcommand} -h".split(),
            check=True,
            stdout=subprocess.PIPE,
        )
        out2 = subprocess.run(
            f"python -m demesdraw {subcommand} --help".split(),
            check=True,
            stdout=subprocess.PIPE,
        )
        assert out1.stdout == out2.stdout

    @pytest.mark.parametrize(
        "params",
        [
            "",
            "--log-time",
            "--aspect 1",
            "--scale 1.2",
            "--aspect 1 --scale 1.2 --log-time",
        ],
    )
    @pytest.mark.parametrize("subcommand", ["tubes", "size_history"])
    def test_output_file_is_created(self, subcommand, params):
        input_file = tests.example_files()[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = pathlib.Path(tmpdir) / "plot.pdf"
            subprocess.run(
                f"python -m demesdraw {subcommand} {params} "
                f"{input_file} {output_file}".split(),
                check=True,
            )
            assert output_file.exists()
            assert output_file.stat().st_size > 0

    def test_tubes_scale_bar(self):
        self.test_output_file_is_created("tubes", "--scale-bar")

    @pytest.mark.parametrize("subcommand", ["tubes", "size_history"])
    def test_input_from_stdin(self, subcommand):
        input_file = tests.example_files()[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = pathlib.Path(tmpdir) / "plot.pdf"
            with open(input_file) as f:
                subprocess.run(
                    f"python -m demesdraw {subcommand} - {output_file}".split(),
                    check=True,
                    stdin=f,
                )
            assert output_file.exists()
            assert output_file.stat().st_size > 0
