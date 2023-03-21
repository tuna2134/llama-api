import sys
import llamacpp

from fastapi import FastAPI


app = FastAPI()
model = None


@app.on_event("startup")
async def llama_start():
    params = llamacpp.gpt_params(
        './ggml-model-q4_0.bin',  # model,
        512,  # ctx_size
        100,  # n_predict
        40,  # top_k
        0.95,  # top_p
        0.85,  # temp
        1.30,  # repeat_penalty
        -1,  # seed
        8,  # threads
        64,  # repeat_last_n
        8,  # batch_size
    )
    model = llamacpp.PyLLAMA(params)


@app.get("/")
async def llama(promps: str):
    model.add_bos()     # Adds "beginning of string" token
    model.update_input("A llama is a")
    model.print_startup_stats()
    model.prepare_context()

    model.ingest_all_pending_input(True)
    content = ""
    while not model.is_finished():
        text, is_finished = model.infer_text()
        content += text

        if is_finished:
            break

    model.print_end_stats()
    return content
