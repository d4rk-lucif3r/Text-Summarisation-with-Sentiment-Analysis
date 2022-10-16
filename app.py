import gradio as gr
from transformers import pipeline

from articles import *


def summarise(text, max_length=130, min_length=30, model='facebook/bart-large-cnn'):
    summarizer = pipeline("summarization", model=model)
    result = summarizer(text, max_length=int(max_length), min_length=int(min_length))
    sent_analysis = pipeline(
        "sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    sent = sent_analysis(result[0]['summary_text'])[0]
    if sent['label'] == 'POS':
        sent1 = 'Sentiment of this Article is Positive with a score of {:.2f}%'.format(
            sent['score']*100)
    elif sent['label'] == 'NEU':
        sent1 = 'Sentiment of this Article is Neutral with a score of {:.2f}%'.format(
            sent['score']*100)
    elif sent['label'] == 'NEG':
        sent1 = 'Sentiment of this Article is Negative with a score of {:.2f}%'.format(
            sent['score']*100)
    return result[0]['summary_text'], sent1


demo = gr.Blocks()

with demo:
    gr.Markdown("# Text Summarizer with Sentiment Analysis")
    gr.Markdown(
        "### Enter the text you want to summarise or choose from examples below")
    with gr.Tabs():
        with gr.TabItem("Examples"):
            with gr.Column():
                rad = gr.components.Radio(
                    ['Article 1', 'Article 2', 'Article 3'], label='Select article and wait till it appears!')
                text1 = gr.Textbox(label='Example')
                rad2 = gr.components.Radio(
                    ['Facebook\'s Large Bart Model', 'DistilBart', 'Google\'s Pegasus'], label='Select Model for summarisation')
                max1 = gr.components.Slider(
                    100, 200, 130, label='Max Length for Summary')
                min1 = gr.components.Slider(
                    20, 100, 30, label='Min Length for Summary')
            submit1 = gr.Button('Submit')
        with gr.TabItem("Do it yourself!"):
            with gr.Column():
                text2 = gr.components.Textbox(
                    label='Enter your own text here!')
                rad3 = gr.components.Radio(
                    ['Facebook\'s Large Bart Model', 'DistilBart', 'Google\'s Pegasus'], label='Select Model for summarisation')
                max2 = gr.components.Slider(
                    0, 200, 130, label='Max Length for Summary')
                min2 = gr.components.Slider(
                    0, 100, 30, label='Min Length for Summary')
            submit2 = gr.Button('Submit')

        def action1(choice):
            if choice == 'Article 1':
                return ARTICLE_1
            elif choice == 'Article 2':
                return ARTICLE_2
            elif choice == 'Article 3':
                return ARTICLE_3

        def models(model_name):
            if model_name == 'Facebook\'s Large Bart Model':
                return 'facebook/bart-large-cnn'
            elif model_name == 'DistilBart':
                return 'sshleifer/distilbart-cnn-12-6'
            elif model_name == 'Google\'s Pegasus':
                return 'google/pegasus-xsum'
            elif model_name is None:
                return 'facebook/bart-large-cnn'

        rad.change(action1, rad, text1)

        op = gr.outputs.Textbox(label='Summary')
        op2 = gr.outputs.Textbox(label='Sentiment Analysis of Summary')

        def fn(text, model, max_length, min_length):
            model = models(model)
            result = summarise(text, max_length, min_length, model)
            return result

        submit1.click(fn=fn, inputs=[text1, rad2,
                      max1, min1], outputs=[op, op2])
        submit2.click(fn=fn, inputs=[text2, rad3,
                      max2, min2], outputs=[op, op2])
        gr.Markdown("### Made with ❤️ by Arsh using TrueFoundry's Gradio Deployment")
        gr.Markdown(
            "### [Github Repo](https://github.com/d4rk-lucif3r/Text-Summarisation-with-Sentiment-Analysis)")
        gr.Markdown(
            '### [Blog](https://lucif3r4.medium.com/summarizing-text-with-transformers-and-deploying-it-as-gradio-app-d96cc11cbf01)')
demo.queue()
demo.launch(server_port=8080, server_name='0.0.0.0') # Launch the gradio block
