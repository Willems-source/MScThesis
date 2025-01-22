

import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from matplotlib import pyplot as plt

# Custom modules
from src.data.preprocessing import split_into_chunks
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics

############################################################################
# YOUR 50 TOPICS + COLORS
############################################################################
MY_TOPIC_LABELS = [
    #"General News on Trump & Presidency",
    "Trump, Mueller Investigation & Russia",
    "US Politics: McConnell, Cruz & Nominees",
    "Tax Reform and Congressional Debates",
    "Israeli-Palestinian Conflict",
    "NFL Protests and Anthem Controversy",
    "Gun Control & Firearm Legislation",
    "Immigration Policies and Deportation",
    "Kurdish Issues in Iraq & Syria",
    "North Korea: Pyongyang & Missiles",
    "Obamacare & Healthcare Repeal",
    "LGBT Rights and Abortion Policies",
    "Rohingya Refugees in Myanmar",
    "Iran Nuclear Sanctions & Diplomacy",
    "Puerto Rico: Hurricane Recovery",
    "Catalonia Independence & Spain Politics",
    "Venezuelan Politics & Chavez Legacy",
    "Middle-East Militants & Troop Activity",
    "German Politics: Merkel & Coalitions",
    "Brexit Negotiations & EU Relations",
    "Russia: Putin & Sanctions",
    "Zimbabwe Politics: Mugabe's Leadership",
    "Terrorism, Suspects & Countermeasures",
    "Illinois Budget & Fiscal Policies",
    "Refugees & Migration Challenges",
    "FCC & Net Neutrality Debate",
    "Turkey Politics: Erdogan's Policies",
    "Chinese Politics: Mao & Zedong Legacy",
    "Media: Radio & Broadcasts",
    "Earthquakes & Disaster Relief",
    "US Elections: Giuliani & Romney",
    "Philippines Politics: Duterte's Leadership",
    "Congress Allegations & Misconduct",
    "Flint Water Crisis & Contamination",
    "New Zealand: Elections & Politics",
    "Japan Politics: Abe & Elections",
    "Hastert Trial: Sentencing & Scandals",
    "Bosnia & Serbian Ethnic Conflict",
    "Pakistan Politics: Sharif Leadership",
    "Canada Politics: Trudeau & Leadership",
    "Activism: Soros & Koch Influence",
    "Bridgegate Scandal: Christie",
    "Marijuana Legalization & Cannabis",
    "Bali Volcanic Eruptions",
    "Saudi Arabia: Women's Rights Reform",
    "JavaScript in Web Contexts",
    "Food & Pie-Related Topics",
    "Naval Issues: Submarines & Sailors",
    "British Royals: Prince Harry & Markle",
    "Rock Musicians & Pop Culture"
]

MY_COLORS_50 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#9edae5", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#c49c94",
    "#c5b0d5", "#ff9896", "#98df8a", "#ffbb78", "#aec7e8",
    "#fdae6b", "#fdd0a2", "#c49c94", "#d7b5d8", "#c5b0d5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7969c", "#ff9896", "#d6616b", "#ad494a",
    "#843c39", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
    "#c7c7c7", "#7f7f7f", "#c49c94", "#ffbb78", "#2ca02c"
]

# Map from topic_id -> label, topic_id -> color
my_custom_labels = { i: MY_TOPIC_LABELS[i] for i in range(len(MY_TOPIC_LABELS)) }
my_color_map = { i: MY_COLORS_50[i] for i in range(len(MY_COLORS_50)) }

############################################################################
# LOAD YOUR BERTopic MODEL AND DATA
############################################################################
topic_model_path = "C:/Thesis/MScThesis/Thesis/topic_models/finetuned_isot_50_topics_model"
filepath = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/isot_metrics_finetuned_isot_50_topics_model_fulldataset.csv"

topic_model = BERTopic.load(topic_model_path)
data = pd.read_csv(filepath)

# HELPER FUNCTIONS
def highlight_words(text, words, color):
    highlighted_text = text
    for w in words:
        highlighted_text = highlighted_text.replace(
            w, f"<span style='background-color:{color}'>{w}</span>"
        )
        cap_w = w.capitalize()
        highlighted_text = highlighted_text.replace(
            cap_w, f"<span style='background-color:{color}'>{cap_w}</span>"
        )
    return highlighted_text

def analyze_full_article(article_text, topic_model, custom_labels, top_n=3):
    _, full_probs = topic_model.transform([article_text])
    full_probs = full_probs[0]
    prob_sum = full_probs.sum()

    top_ids = np.argsort(full_probs)[::-1][:top_n]
    top_topics = []
    for tid in top_ids:
        c_label = custom_labels.get(tid, f"Topic {tid}")
        topic_words = topic_model.get_topic(tid) or []
        rep_words = [w for w, _ in topic_words[:10]]
        top_topics.append({
            "Topic ID": tid,
            "Custom Label": c_label,
            "Probability": full_probs[tid],
            "Representative Words": rep_words
        })
    return top_topics, full_probs, prob_sum

def analyze_chunks(chunks, topic_model, custom_labels, top_n=3):
    chunk_analyses = []
    for chunk_text in chunks:
        _, probs = topic_model.transform([chunk_text])
        probs = probs[0]
        prob_sum = probs.sum()

        top_ids = np.argsort(probs)[::-1][:top_n]
        top_topics = []
        for tid in top_ids:
            c_label = custom_labels.get(tid, f"Topic {tid}")
            topic_words = topic_model.get_topic(tid) or []
            rep_words = [w for w, _ in topic_words[:10]]
            top_topics.append({
                "Topic ID": tid,
                "Custom Label": c_label,
                "Probability": probs[tid],
                "Representative Words": rep_words
            })
        chunk_analyses.append({
            "text": chunk_text,
            "top_topics": top_topics,
            "prob_sum": prob_sum
        })
    return chunk_analyses

############################################################################
# MAIN EXPLAINABILITY -> HTML
############################################################################
def explain_article_coherence_html(
    article_text,
    topic_model,
    custom_labels,
    color_map,
    js_threshold=0.425,
    chunk_size=5,
    top_n=3,
    output_html="article_explanation.html",
    label=None  # Add ground truth label as an argument
):
    # 1) Split
    chunks = split_into_chunks(article_text, sentences_per_chunk=chunk_size, min_chunks=3)

    # 2) Full article analysis
    full_topics, full_probs, full_sum = analyze_full_article(
        article_text, topic_model, custom_labels, top_n
    )

    # 3) JS + classification
    js_values = compute_js_values_for_chunks(chunks=chunks, original_probs=full_probs, model=topic_model)
    metrics_dict = compute_coherence_metrics(js_values)
    coherence_rmse = metrics_dict["RMSE"]
    classification = "real" if coherence_rmse < js_threshold else "fake"

    # 4) Analyze chunks
    chunk_analyses = analyze_chunks(chunks, topic_model, custom_labels, top_n)

    # 5) Highlight the full article with each topic color
    full_article_highlighted = article_text
    for tinfo in full_topics:
        tid = tinfo["Topic ID"]
        color = color_map.get(tid, "#ffffff")  # fallback white
        rep_words = tinfo["Representative Words"]
        full_article_highlighted = highlight_words(full_article_highlighted, rep_words, color)

    # 6) Build HTML
    html_parts = []
    html_parts.append("<html><head><meta charset='utf-8'>")
    html_parts.append("<title>Article Explanation</title>")
    html_parts.append("""
    <style>
      body { font-family: Arial, sans-serif; margin: 10px; }
      .container { display: flex; margin-bottom: 2em; }
      .left-pane { flex: 1; padding: 1em; border-right: 2px solid #ccc; }
      .right-pane { width: 250px; padding: 1em; font-size: 0.9em; }
      .chunk-box { margin-bottom: 2em; }
      pre { white-space: pre-wrap; }
      table { border-collapse: collapse; }
      td, th { border: 1px solid #ccc; padding: 5px; }
      h2, h3 { margin-top: 1em; }
      .topic-box { margin-bottom:1em; }
    </style>
    </head><body>
    """)

    # Classification summary
    is_correct = (classification == label.lower())
    correctness = "correctly" if is_correct else "wrongly"

    html_parts.append(f"<h1>Classification: {classification.upper()}</h1>")
    html_parts.append(
        f"<p>Coherence RMSE = {coherence_rmse:.3f}, threshold={js_threshold}.</p>"
        f"<p>Based on the RMSE value of <strong>{coherence_rmse:.3f}</strong>, which is "
        f"<strong>{'higher' if coherence_rmse > js_threshold else 'lower'}</strong> than the threshold of {js_threshold}, "
        f"the article is <strong>{correctness}</strong> classified as <strong>{classification.upper()}</strong> "
        f"(Ground Truth: {label.upper()})."
    )
    html_parts.append("<hr>")


    # Compute RMSE directly from JS values
    rmse_computed = np.sqrt(np.mean(np.array(js_values) ** 2))
    html_parts.append("<hr>")

    # Full article container
    html_parts.append("<h2>Full Article</h2>")
    html_parts.append("<div class='container'>")
    # Left -> article text
    html_parts.append("<div class='left-pane'>")
    html_parts.append("<h3>Text</h3>")
    html_parts.append(f"<pre>{full_article_highlighted}</pre>")
    html_parts.append("</div>")

    # Right -> top topics
    html_parts.append("<div class='right-pane'>")
    html_parts.append("<h3>Top Topics (Full Article)</h3>")
    for tinfo in full_topics:
        tid = tinfo["Topic ID"]
        color = color_map.get(tid, "#ffffff")
        html_parts.append("<div class='topic-box'>")
        html_parts.append(f"<strong style='color:{color}'>{tinfo['Custom Label']}</strong><br>")
        html_parts.append(f"Topic ID: {tid}<br>")
        html_parts.append(f"Probability: {tinfo['Probability']:.3f}<br>")
        html_parts.append(f"Keywords: {', '.join(tinfo['Representative Words'])}")
        html_parts.append("</div>")
    html_parts.append(f"<p>Sum of probabilities: {full_sum:.3f}</p>")
    html_parts.append("</div>")  # end right-pane
    html_parts.append("</div>")  # end container
    # Now each chunk
    html_parts.append("<h2>Chunks</h2>")

    for i, cinfo in enumerate(chunk_analyses, start=1):
        chunk_text = cinfo["text"]
        top_topics = cinfo["top_topics"]
        prob_sum = cinfo["prob_sum"]

        # highlight chunk text with each topic color
        chunk_highlighted = chunk_text
        for tinfo in top_topics:
            tid = tinfo["Topic ID"]
            color = color_map.get(tid, "#ffffff")
            rep_words = tinfo["Representative Words"]
            chunk_highlighted = highlight_words(chunk_highlighted, rep_words, color)

        html_parts.append("<div class='chunk-box container'>")
        # Left -> chunk text
        html_parts.append("<div class='left-pane'>")
        html_parts.append(f"<h3>Chunk {i}</h3>")
        html_parts.append(f"<pre>{chunk_highlighted}</pre>")
        html_parts.append("</div>")

        # Right -> chunk topics
        html_parts.append("<div class='right-pane'>")
        html_parts.append(f"<h3>Top Topics (Chunk {i})</h3>")
        for tinfo in top_topics:
            tid = tinfo["Topic ID"]
            color = color_map.get(tid, "#ffffff")
            html_parts.append("<div class='topic-box'>")
            html_parts.append(f"<strong style='color:{color}'>{tinfo['Custom Label']}</strong><br>")
            html_parts.append(f"Topic ID: {tid}<br>")
            html_parts.append(f"Probability: {tinfo['Probability']:.3f}<br>")
            html_parts.append(f"Keywords: {', '.join(tinfo['Representative Words'])}")
            html_parts.append("</div>")
        html_parts.append(f"<p><i>Sum of probabilities: {prob_sum:.3f}</i></p>")
        html_parts.append("</div>")
        html_parts.append("</div>")  # chunk-box container

    # JS Divergence table
    html_parts.append("<hr>")
    html_parts.append("<h2>Jensen-Shannon Divergence per Chunk</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Chunk</th><th>JS Divergence</th></tr>")
    for i, val in enumerate(js_values, start=1):
        html_parts.append(f"<tr><td>{i}</td><td>{val:.3f}</td></tr>")
    html_parts.append("</table>")

    # Compute RMSE directly from JS values
    rmse_computed = np.sqrt(np.mean(np.array(js_values) ** 2))

    # RMSE calculation / explanation
    html_parts.append("<hr>")
    html_parts.append("<p><strong>Given the JS divergence values:</strong></p>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>JS Divergences: {', '.join([f'{v:.3f}' for v in js_values])}</li>")
    html_parts.append("</ul>")
    html_parts.append("<p>")
    html_parts.append(
        f"RMSE is calculated as: <strong>√(mean([{ ' + '.join([f'({v:.3f}²)' for v in js_values]) }]))</strong><br>"
    )
    html_parts.append(f"Which evaluates to: <strong>{rmse_computed:.3f}</strong>.")
    html_parts.append("</p>")


    # Final classification line
    comparison_word = "lower" if coherence_rmse < js_threshold else "higher"
    html_parts.append("<p>")
    html_parts.append(
        f"Based on the RMSE value of <strong>{coherence_rmse:.3f}</strong>, "
        f"which is <strong>{comparison_word}</strong> than the threshold of {js_threshold}, "
        f"the article is classified as <strong>{classification.upper()}</strong>."
    )
    html_parts.append("</p>")

    html_parts.append("</body></html>")

    final_html = "\n".join(html_parts)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"[INFO] Explanation saved to {output_html}")
    print(f"Classification: {classification.upper()} (RMSE={coherence_rmse:.3f})")

    return {
        "classification": classification,
        "coherence_rmse": coherence_rmse,
        "chunk_analyses": chunk_analyses,
        "js_values": js_values,
        "output_html": output_html
    }

if __name__ == "__main__":
    sample_row = data.iloc[15445]
    #article_text = sample_row["text"]
    article_text = "South Korean nuclear experts, checking for contamination after North Korea's sixth and largest nuclear test, said on Friday they have found minute traces of radioactive xenon gas but that it was too early to specify its source. The Nuclear Safety and Security Commission (NSSC) said it had been conducting tests on land, air and water samples since shortly after North Korea's nuclear test on Sunday. The statement said the commission was analyzing how the xenon entered South Korean territory and will make a decision at a later time whether the material is linked to North Korea's nuclear test. A U.S. appeals court in Washington on Tuesday upheld a lower court's decision to allow President Donald Trump's commission investigating voter fraud to request data on voter rolls from U.S. states. But the NSSC said it had detected xenon-133, a radioactive isotope that does not occur naturally and which has in the past been linked to North Korea's nuclear tests. Catalonia is back to square one, said Marco Protopapa, an analyst at JP Morgan, forecasting that tensions would quickly return between Madrid and an emboldened pro-independence camp eager to exploit the tactical advantage of a favorable election outcome."
    #article_text = "South Korean nuclear experts, checking for contamination after North Korea's sixth and largest nuclear test, said on Friday they have found minute traces of radioactive xenon gas but that it was too early to specify its source. The Nuclear Safety and Security Commission (NSSC) said it had been conducting tests on land, air and water samples since shortly after North Korea's nuclear test on Sunday. The statement said the commission was analyzing how the xenon entered South Korean territory and will make a decision at a later time whether the material is linked to North Korea's nuclear test. Xenon is a naturally occurring, colorless gas that is used in manufacturing of some sorts of lights. But the NSSC said it had detected xenon-133, a radioactive isotope that does not occur naturally and which has in the past been linked to North Korea's nuclear tests. There was no chance the xenon will have an impact on South Korea's territory or population, the statement said."
    #article_text = "South Korean nuclear experts, checking for contamination after North Korea's sixth and largest nuclear test, said on Friday they have found minute traces of radioactive xenon gas but that it was too early to specify its source. The Nuclear Safety and Security Commission (NSSC) said it had been conducting tests on land, air and water samples since shortly after North Korea s nuclear test on Sunday. The statement said the commission was analyzing how the xenon entered South Korean territory and will make a decision at a later time whether the material is linked to North Korea s nuclear test. Xenon is a naturally occurring, colorless gas that is used in manufacturing of some sorts of lights. But the NSSC said it had detected xenon-133, a radioactive isotope that does not occur naturally and which has in the past been linked to North Korea s nuclear tests. There was no chance the xenon will have an impact on South Korea s territory or population, the statement said."
    #sample_row = data.sample(1).iloc[0]
    #article_text = sample_row["text"]
    label = sample_row["label"]  # Ground truth

    print(f"Actual label from dataset: {label}")

    results = explain_article_coherence_html(
        article_text=article_text,
        topic_model=topic_model,
        custom_labels=my_custom_labels,
        color_map=my_color_map,
        js_threshold=0.425,
        chunk_size=5,
        top_n=3,
        output_html="C:/Thesis/MScThesis/gc_experiments_results/explainability/replacementexample_article_explanation_minchunk3_chunksize5.html",
        label=label  # Pass ground truth
    )
    print(f"HTML file is here: {results['output_html']}")


