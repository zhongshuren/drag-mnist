import json
from fastapi import WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }

def wrapper(app, content_manager):
    with open('app/index.html', encoding='utf-8') as f:
        html = f.read()

    @app.get('/')
    async def get():
        return HTMLResponse(html, headers=headers)

    app.mount('/static', StaticFiles(directory="app/static"), name="static")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            if not content_manager.generate_input:
                content_manager.update_state(json.loads(data))

    @app.get('/start-video')
    async def video_feed():
        return StreamingResponse(content_manager.get_frame(),
                                 media_type='multipart/x-mixed-replace; boundary=frame')

    @app.get('/reset')
    async def reset_feed():
        content_manager.reset_state()
        return {}

    return app