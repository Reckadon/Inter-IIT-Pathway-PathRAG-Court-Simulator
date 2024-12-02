import pathway as pw

from .endpoints import process_case

def create_app():
    app = pw()
    
    # Configure app settings
    app.config.update({
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "workers": 4
    })
    
    # Register endpoints
    app.add_endpoint(process_case)
    # app.add_endpoint(search_legal_data)
    
    return app

def main():
    app = create_app()
    app.run()

if __name__ == "__main__":
    main() 