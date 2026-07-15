

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates




app = FastAPI(title="Linear Regression",
              description="Linear Regression via gradient descent built from stratch on weather data to predict the weather",
            )

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(req: Request ):
    return templates.TemplateResponse(name="index.html", request = req  )

