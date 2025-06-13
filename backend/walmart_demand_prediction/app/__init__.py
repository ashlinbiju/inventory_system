from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import and register your routes
    from app.routes import routes
    app.register_blueprint(routes)

    return app
