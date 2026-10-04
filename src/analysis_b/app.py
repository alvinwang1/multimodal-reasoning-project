"""Staged-reveal annotation tool (Analysis B).

Stage 0 shows the question, options, image caption and audio transcript only. The annotator
says whether text is enough, then chooses to reveal the image, reveal the audio, or answer.
Every step is logged in order. Each finished item is appended to results/analysis_b/<name>.jsonl;
restarting skips finished items (an item that was interrupted midway restarts from stage 0).

    python -m src.data && python -m src.analysis_b.assign   # once, to create assignments + media
    python -m src.analysis_b.app                             # http://127.0.0.1:7860
    python -m src.analysis_b.app --annotator arjun --port 7861
"""
import argparse
import json
import time
from datetime import datetime, timezone

import gradio as gr

from src.common import append_jsonl, load_config, read_jsonl, repo_path
from src.data import LETTERS

APP_VERSION = 1
CONF = ["1", "2", "3", "4", "5"]
NO_ANSWER = "(no answer)"


class Store:
    def __init__(self, cfg):
        bcfg = cfg["analysis_b"]
        self.media = repo_path(bcfg["media_dir"])
        self.results = repo_path(bcfg["results_dir"])
        assign = json.loads(repo_path(bcfg["assignments"]).read_text())
        self.per_annotator = assign["per_annotator"]
        self.shared = set(assign["shared"])
        self.items = {int(k): v for k, v in json.loads((self.media / "items.json").read_text()).items()}

    def path(self, annotator):
        return self.results / f"{annotator}.jsonl"

    def remaining(self, annotator):
        done = {r["id"] for r in read_jsonl(self.path(annotator))}
        return [i for i in self.per_annotator[annotator] if i not in done]


def item_markdown(item):
    opts = "\n".join(f"- **{LETTERS[j]}.** {o}" for j, o in enumerate(item["options"]))
    return (f"### {item['question']}\n\n{opts}\n\n"
            f"**Image caption:** {item['caption']}\n\n"
            f"**Audio transcript / description:** {item['transcript']}")


def build_app(store, default_annotator=None):
    annotators = list(store.per_annotator)

    with gr.Blocks(title="OmniBench staged reveal") as demo:
        state = gr.State({"phase": "idle"})
        gr.Markdown("## OmniBench staged-reveal annotation\n"
                    "Answer using as little input as you need. Do not look items up elsewhere.")
        with gr.Row():
            who = gr.Dropdown(annotators, value=default_annotator, label="Annotator")
            start = gr.Button("Start / resume", variant="primary")
        progress = gr.Markdown()
        status = gr.Markdown()
        item_md = gr.Markdown(visible=False)

        with gr.Group(visible=False) as g0:
            gr.Markdown("**Stage 0: text only**")
            text_enough = gr.Radio(["yes", "no"], label="Is text enough to answer?")
            conf0 = gr.Radio(CONF, label="Confidence (1 = guess, 5 = certain)")
            ans0 = gr.Radio(LETTERS + [NO_ANSWER], value=NO_ANSWER, label="Answer now (optional)")
            submit0 = gr.Button("Submit stage 0", variant="primary")

        with gr.Group(visible=False) as gact:
            gr.Markdown("**Next action**")
            with gr.Row():
                b_img = gr.Button("Reveal image")
                b_aud = gr.Button("Reveal audio")
                b_ans = gr.Button("Answer now", variant="primary")

        image = gr.Image(visible=False, type="filepath", label="Image", interactive=False)
        audio = gr.Audio(visible=False, type="filepath", label="Audio", interactive=False)

        with gr.Group(visible=False) as gfin:
            gr.Markdown("**Final answer**")
            final = gr.Radio(LETTERS, label="Answer")
            conf_f = gr.Radio(CONF, label="Confidence (1 = guess, 5 = certain)")
            note = gr.Textbox(label="Which input decided it? (e.g. 'caption mentions a guitar', "
                                    "'needed audio to hear the dog')", lines=2)
            submit_f = gr.Button("Submit and next", variant="primary")

        outputs = [state, progress, status, item_md, g0, gact, b_img, b_aud, image, audio, gfin,
                   text_enough, conf0, ans0, final, conf_f, note]

        def render(st, msg="", reset_inputs=False):
            phase = st["phase"]
            cur = st.get("cur")
            item = store.items[cur["id"]] if cur else None
            if st.get("annotator"):
                total = len(store.per_annotator[st["annotator"]])
                left = len(st.get("queue", [])) + (1 if cur else 0)
                prog = f"**{st['annotator']}**: item {total - left + (1 if cur else 0)} / {total}"
                if phase == "done":
                    prog = f"**{st['annotator']}**: all {total} items done. Thank you!"
            else:
                prog = "Pick your name and press **Start / resume**."
            rev = cur["revealed"] if cur else {"image": False, "audio": False}
            keep = gr.update()
            return [
                st, prog, msg,
                gr.update(visible=item is not None and phase != "done",
                          value=item_markdown(item) if item else ""),
                gr.update(visible=phase == "stage0"),
                gr.update(visible=phase == "action"),
                gr.update(interactive=not rev["image"]),
                gr.update(interactive=not rev["audio"]),
                gr.update(visible=bool(item) and rev["image"],
                          value=str(store.media / item["image"]) if item and rev["image"] else None),
                gr.update(visible=bool(item) and rev["audio"],
                          value=str(store.media / item["audio"]) if item and rev["audio"] else None),
                gr.update(visible=phase == "answer"),
                *([gr.update(value=None), gr.update(value=None), gr.update(value=NO_ANSWER),
                   gr.update(value=None), gr.update(value=None), gr.update(value="")]
                  if reset_inputs else [keep] * 6),
            ]

        def next_item(st):
            if not st["queue"]:
                st["phase"], st["cur"] = "done", None
                return st
            i = st["queue"].pop(0)
            st["cur"] = {"id": i, "t0": time.time(),
                         "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "steps": [], "revealed": {"image": False, "audio": False}, "stage0": None}
            st["phase"] = "stage0"
            return st

        def on_start(name, st):
            if not name:
                return render(st, "⚠️ Pick your name first.")
            st = {"annotator": name, "queue": store.remaining(name), "cur": None}
            return render(next_item(st), reset_inputs=True)

        def on_stage0(st, te, c0, a0):
            if te is None or c0 is None:
                return render(st, "⚠️ Answer *Is text enough?* and confidence first.")
            cur = st["cur"]
            cur["stage0"] = {"text_enough": te == "yes", "conf": int(c0),
                             "answer_idx": None if a0 in (None, NO_ANSWER) else LETTERS.index(a0),
                             "t": round(time.time() - cur["t0"], 2)}
            st["phase"] = "action"
            return render(st)

        def on_action(kind):
            def handler(st):
                cur = st["cur"]
                cur["steps"].append({"action": kind, "t": round(time.time() - cur["t0"], 2)})
                if kind == "answer":
                    st["phase"] = "answer"
                else:
                    cur["revealed"][kind.split("_")[1]] = True
                    if all(cur["revealed"].values()):
                        cur["steps"].append({"action": "answer", "t": round(time.time() - cur["t0"], 2)})
                        st["phase"] = "answer"
                return render(st)
            return handler

        def on_final(st, ans, cf, nt):
            if ans is None or cf is None:
                return render(st, "⚠️ Choose an answer and a confidence.")
            cur, i = st["cur"], st["cur"]["id"]
            item = store.items[i]
            append_jsonl(store.path(st["annotator"]), {
                "id": i, "annotator": st["annotator"], "shared": i in store.shared,
                "audio_type": item["audio_type"], "task_type": item["task_type"],
                "stage0": cur["stage0"], "steps": cur["steps"], "revealed": cur["revealed"],
                "final": {"answer_idx": LETTERS.index(ans), "conf": int(cf), "note": (nt or "").strip()},
                "n_reveals": sum(cur["revealed"].values()),
                "duration_s": round(time.time() - cur["t0"], 2), "started_at": cur["started_at"],
                "app_version": APP_VERSION,
            })
            return render(next_item(st), "✅ Saved.", reset_inputs=True)

        start.click(on_start, [who, state], outputs)
        submit0.click(on_stage0, [state, text_enough, conf0, ans0], outputs)
        b_img.click(on_action("reveal_image"), [state], outputs)
        b_aud.click(on_action("reveal_audio"), [state], outputs)
        b_ans.click(on_action("answer"), [state], outputs)
        submit_f.click(on_final, [state, final, conf_f, note], outputs)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--annotator", default=None, help="preselect a name in the dropdown")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="create a temporary public Gradio link so teammates can annotate remotely")
    args = ap.parse_args()
    store = Store(load_config(args.config))
    build_app(store, args.annotator).launch(server_port=args.port, share=args.share,
                                            allowed_paths=[str(store.media)])


if __name__ == "__main__":
    main()
