import gradio as gr
from vector_store import search_top_k, add_document

def search_interface(query, metric):
    results = search_top_k(query, top_k=5, metric=metric)
    display = ""
    for result in results:
        display += f"🔹 Rank {result['rank']} (Score: {result['score']:.4f})\n"
        display += f"{result['text']}\n\n"
    return display

def add_interface(new_text):
    result = add_document(new_text)
    return f"✅ {result['message']} (Document ID: {result['id']})"

# Gradio Blocks UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Document Similarity Search API")

    with gr.Tab("🔍 Search"):
        query = gr.Textbox(label="Enter your query")
        metric = gr.Radio(["l2", "cosine", "dot"], label="Similarity Metric", value="l2")
        search_btn = gr.Button("Search")
        output = gr.Textbox(label="Top 5 Similar Documents")
        search_btn.click(fn=search_interface, inputs=[query, metric], outputs=output)

    with gr.Tab("➕ Add Document"):
        new_doc = gr.Textbox(label="New Document Text")
        add_btn = gr.Button("Add")
        add_status = gr.Textbox(label="Status")
        add_btn.click(fn=add_interface, inputs=new_doc, outputs=add_status)

demo.launch()
