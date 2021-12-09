"""
Add version string to the navbar and footer.
"""
import demesdraw


def inject_version(app, config):
    v = demesdraw.__version__
    if v != "undefined":
        v_short = v.split("+")[0]
        config.html_theme_options["extra_navbar"] = f"demesdraw {v_short}"
        config.html_theme_options["extra_footer"] = f"demesdraw {v}"


def setup(app):
    app.connect("config-inited", inject_version)
