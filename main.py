from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from gemini_agentic import run_ai_stream


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print("Request for index page received")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    file_name = "favicon.ico"
    file_path = "./static/" + file_name
    return FileResponse(
        path=file_path, headers={"mimetype": "image/vnd.microsoft.icon"}
    )


@app.post("/hello", response_class=HTMLResponse)
async def hello(request: Request, name: str = Form(...)):
    if name:
        print("Request for hello page received with name=%s" % name)
        return templates.TemplateResponse(
            "hello.html", {"request": request, "name": name}
        )
    else:
        print(
            "Request for hello page received with no name or blank name -- redirecting"
        )
        return RedirectResponse(
            request.url_for("index"), status_code=status.HTTP_302_FOUND
        )


@app.post("/askai", response_class=HTMLResponse)
async def askai(request: Request, name: str = Form(...),show_log_data:bool =Form(False)):
    if name:
        print("Request for hello page received with name=%s" % name)
        if show_log_data:
            name,log = run_ai_stream(name, 1,True)
            lines = log.split('\n')
            html_string = '<br>'.join(lines)
            return templates.TemplateResponse(
                "airesponse.html", {"request": request, "name": name,"reasoning_steps":html_string}
            )
        else:
            name,log = run_ai_stream(name, 1,True)
            print("answer:", name)
            return templates.TemplateResponse(
                "airesponse.html", {"request": request, "name": name,"reasoning_steps":"Not included"}
            )

    else:
        print(
            "Request for hello page received with no name or blank name -- redirecting"
        )
        return RedirectResponse(
            request.url_for("index"), status_code=status.HTTP_302_FOUND
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
