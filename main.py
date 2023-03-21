import sys
import llamacpp

from fastapi import FastAPI


app = FastAPI()


params = llamacpp.gpt_params(
    './ggml-alpaca-7b-q4.bin',  # model,
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


@app.get("/")
async def llama(promps: str):
    model = llamacpp.PyLLAMA(params)
    model.add_bos()     # Adds "beginning of string" token
    model.update_input(promps)
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
    return {"content": content}
