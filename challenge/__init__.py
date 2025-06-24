def get_app():
    """Lazy loading to avoid heavy imports during package import."""
    from challenge.api import app
    return app

# Keep compatibility for deployment
application = get_app()