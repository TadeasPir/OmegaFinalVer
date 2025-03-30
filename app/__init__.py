from flask import Flask
from config.settings import Config


def create_app(config_class=Config):
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')
    app.config.from_object(config_class)

    # Register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    return app
