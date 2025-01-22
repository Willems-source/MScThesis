import pandas as pd
from bertopic import BERTopic
from src.data.preprocessing import split_into_chunks
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics
import matplotlib.pyplot as plt

# Load the pre-trained BERTopic model
#topic_model = BERTopic.load("C:/Thesis/MScThesis/Thesis/topic_models/finetuned_isot_50_topics_model")
#topic_model = BERTopic.load("C:/Thesis/MScThesis/Thesis/topic_models/wsj_topic_model")
topic_model = BERTopic.load("C:/Thesis/MScThesis/Thesis/topic_models/general_50_topic_model")
#topic_model = BERTopic.load("C:/Thesis/MScThesis/Thesis/topic_models/isot_topic_model") # ALL 353 TOPICS

article_text = """
President Biden addressed the challenges of climate change, emphasizing renewable energy and environmental protection repeatedly across several speeches and interviews, while advocating for stronger international cooperation on environmental issues.
LeBron James showcased his exceptional skills during a high-stakes basketball game, repeatedly demonstrating strategic plays and leadership on the court, captivating fans with his consistent performance and dedication to the sport.
President Biden unveiled a comprehensive healthcare reform plan, repeatedly highlighting the benefits for middle-class families and stressing the need for bipartisan support, as he navigated complex policy discussions.
LeBron James engaged in community outreach, repeatedly visiting schools and organizing youth programs, underscoring his commitment to education and social responsibility off the basketball court.
President Biden discussed economic growth strategies, repeatedly citing job creation, infrastructure improvements, and innovative technologies as key drivers of future prosperity in multiple press conferences.
LeBron James participated in philanthropic events, repeatedly donating to charities and speaking on behalf of underprivileged communities, demonstrating his ongoing commitment to giving back.
"""
# Parameters for chunking
sentences_per_chunk = 5
min_chunks = 3

# Split article into chunks
chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)

# 1) Print the generated chunks for verification
if not chunks:
    print("No chunks generated. Check that the article has enough sentences.")
else:
    print("\n========== CHUNKS ==========\n")
    for i, ch in enumerate(chunks, start=1):
        print(f"Chunk {i}:\n{ch}\n")

# 2) Get the full distribution (of length 50) for the entire article
_, article_probs = topic_model.transform([article_text])
article_probs = article_probs[0]  # shape is (50,) if you have 50 topics

# 3) Get the full distribution for each chunk
chunk_probs_list = []
for i, chunk_text in enumerate(chunks, start=1):
    _, probs = topic_model.transform([chunk_text])
    chunk_probs_list.append(probs[0])

# # 4) Print full distributions (all 50 topics) for article and chunks
# print("\n========== FULL TOPIC PROBABILITIES ==========")
# print("For the full article:")
# for topic_id, prob in enumerate(article_probs):
#     print(f"   Topic {topic_id}: {prob:.4f}")

# for i, chunk_probs in enumerate(chunk_probs_list, start=1):
#     print(f"\nFor Chunk {i}:")
#     for topic_id, prob in enumerate(chunk_probs):
#         print(f"   Topic {topic_id}: {prob:.4f}")

# 5) (Optional) Convert distributions to a DataFrame for easy saving or inspection
df_data = {
    "FullArticle": article_probs
}
for i, cp in enumerate(chunk_probs_list, start=1):
    df_data[f"Chunk_{i}"] = cp

df = pd.DataFrame(df_data)
print("\nDataFrame of all topic probabilities (first few rows shown here):")
print(df.head())

# If you only want to see the top-N topics in your printout, do so below:
TOP_N = 3  # or any number you like

# Sort probabilities for the full article, descending
sorted_article = sorted(enumerate(article_probs), key=lambda x: x[1], reverse=True)
topN_article = sorted_article[:TOP_N]

print(f"\n========== TOP {TOP_N} TOPICS: FULL ARTICLE ==========")
for rank, (tid, prob) in enumerate(topN_article, start=1):
    print(f"Rank {rank}: Topic {tid}, Probability={prob:.4f}")

# Sort and print top topics for each chunk
for i, chunk_probs in enumerate(chunk_probs_list, start=1):
    sorted_chunk = sorted(enumerate(chunk_probs), key=lambda x: x[1], reverse=True)
    topN_chunk = sorted_chunk[:TOP_N]
    print(f"\n========== TOP {TOP_N} TOPICS: CHUNK {i} ==========")
    for rank, (tid, prob) in enumerate(topN_chunk, start=1):
        print(f"Rank {rank}: Topic {tid}, Probability={prob:.4f}")

# 6) (Optional) Plot or compute any other metrics as desired
js_values = compute_js_values_for_chunks(chunks, article_probs, topic_model)
rmse = compute_coherence_metrics(js_values)["RMSE"]
print(f"\nJensen–Shannon divergences per chunk: {js_values}")
print(f"Coherence RMSE: {rmse:.4f}")

# #### --------------------------------------- ANALYSIS OF FULL ARTICLE ONLY (ONLY EXTRACTING TOPICS) ------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# from bertopic import BERTopic
# import matplotlib.pyplot as plt

# # Load the pre-trained BERTopic model
# topic_model = BERTopic.load("C:/Thesis/MScThesis/Thesis/topic_models/finetuned_isot_50_topics_model")

# # Define the article text
# article_text = """A Zimbabwean court on Friday acquitted a pastor critical of President Robert Mugabe on charges of committing public violence and disorderly conduct. The charge was linked to an address Evan Mawarire made during a demonstration by university students earlier this year. Mawarire, who is also on trial for separate and more serious charges that he attempted to subvert the government, had been arrested in June after addressing medical students protesting against an increase in fees. Magistrate Tilda Mazhande said state prosecutors had failed to establish a case against the 40-year-old. Mawarire s High Court trial on subversion charges came to a close on Friday after eight state witnesses testified. His lawyer Harrison Nkomo said he would apply to the judge to have the case dismissed. Meanwhile, police fired teargas during sporadic clashes with a group of protesters demonstrating against cash shortages and economic difficulties in central Harare. Pressure group  called for the demonstration following shortages of fuel and panic buying of basic goods last weekend, which the government blamed on social media for spreading false rumors. A reporter and photographer from the private Daily News newspaper covering the protest were beaten with batons and injured by police, Editor-in-Chief Hama Saburi told Reuters. The two were admitted at a local private hospital. Police spokeswoman Charity Charamba could not be reached for comment. A Reuters witness also saw police firing into the air near an office mall in the central business district frequented by illegal foreign currency traders, forcing people to flee. Finance Minister Patrick Chinamasa on Thursday told parliament that the government published rules allowing police to arrest unlicensed foreign currency traders and those found guilty would face up to ten years in prison. Most spots usually occupied by foreign currency dealers were deserted on Friday."""

# # Analyze the full article
# _, article_topic_probs = topic_model.transform([article_text])
# article_topic_probs = article_topic_probs[0]

# # Get the top 3 topics for the full article
# top_topics = sorted(
#     [(i, prob) for i, prob in enumerate(article_topic_probs)],
#     key=lambda x: x[1],
#     reverse=True
# )[:3]

# # Assign consistent colors to the top 3 topics
# colors = plt.cm.tab10.colors  # Use tab10 colormap
# topic_color_map = {topic: colors[i % len(colors)] for i, (topic, _) in enumerate(top_topics)}

# # Plot the top 3 topics
# def plot_full_article_topics(top_topics, topic_color_map):
#     """
#     Plots the top 3 topics for the full article with consistent colors.
#     """
#     labels = [f"Topic {topic}" for topic, _ in top_topics]
#     probabilities = [prob for _, prob in top_topics]
#     colors = [topic_color_map[topic] for topic, _ in top_topics]

#     plt.figure(figsize=(8, 5))
#     plt.bar(labels, probabilities, color=colors)
#     plt.ylabel("Topic Probability")
#     plt.title("Top 3 Topics for Full Article")
#     plt.tight_layout()
#     plt.show()

# # Display the top topics
# print("Top 3 Topics for the Full Article:")
# for rank, (topic, prob) in enumerate(top_topics):
#     topic_words = topic_model.get_topic(topic) or []
#     representative_words = [word for word, _ in topic_words[:10]]
#     print(f"  Rank {rank + 1}:")
#     print(f"    Topic ID: {topic}")
#     print(f"    Probability: {prob:.4f}")
#     print(f"    Representative Words: {', '.join(representative_words)}")

# # Plot the top topics
# plot_full_article_topics(top_topics, topic_color_map)



# # ###COSINE SIMILARITY TEST
# # import pandas as pd
# # from bertopic import BERTopic
# # from src.data.preprocessing import split_into_chunks
# # from src.utils.helper import compute_coherence_metrics
# # from sklearn.metrics.pairwise import cosine_similarity
# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Load the pre-trained BERTopic model
# # topic_model = BERTopic.load("isot_topic_model")

# # # Define the article text
# # article_text = """Prominent Republican senators on Thursday embraced a push to overhaul rules for addressing sexual harassment in the U.S. Congress, signing on to a bill that would protect victims and require lawmakers to pay for their own settlements. The legislation builds on demands to lift the veil of secrecy around sexual harassment and misconduct on Capitol Hill, and has gained steam in recent months as a wave of women have come forward with accusations against prominent American men in politics, media and entertainment. The bipartisan push signaled momentum in the Republican-led U.S. Congress for overhauling a process for handling misconduct allegations that many lawmakers say is antiquated and stacked against victims. The Senate bill, called the Congressional Harassment Reform Act, draws from proposals that Senator Kirsten Gillibrand and Representative Jackie Speier, both Democrats, have been developing. "Congress is really behind the eight-ball. I think that, in many respects, the private sector has acted more swiftly than we have in terms of addressing sexual harassment," Speier said in an interview. High-profile Republican senators co-sponsoring the bill include John Cornyn, the Senate's No. 2 Republican; Ted Cruz; Joni Ernst and Lisa Murkowski. The legislation would require any member of Congress found liable for harassment to pay settlements themselves, rather than with taxpayer funds, as the current process allows. "Congress is not above the laws, and secret settlements with taxpayer money to cover up harassment should no longer be tolerated," Cruz said in a statement. Settlements would be made public automatically unless victims choose to keep them private. Outrage over sexual misconduct in politics helped to fuel an upset victory by Democrat Doug Jones in the U.S. Senate race in deeply conservative Alabama on Tuesday. Voters rejected the Republican candidate in the race, Roy Moore, who had been accused by multiple women of pursuing them when they were teenagers and he was in his 30s, including one woman who said he tried to initiate sexual contact with her when she was 14. Moore denied the allegations but many prominent Republicans distanced themselves from Moore, although President Donald Trump backed him. In Washington, allegations of sexual misconduct prompted the resignations last week of three lawmakers - Democratic Senator Al Franken, Democratic Representative John Conyers and Republican Representative Trent Franks. On Tuesday, Republican Representative Blake Farenthold said he would not seek re-election in November. Politico reported that the congressional Office of Compliance had paid $84,000 from a public fund on behalf of Farenthold to settle a sexual harassment claim in 2015. Reuters has been unable to verify the allegations against Farenthold, who has said that the charges were false and has denied wrongdoing. The 1995 law governing the process for complaints in Congress - created in the wake of a harassment scandal - has been criticized as ineffective. The lengthy and cumbersome process requires victims to go through mandatory mediation and requires complete secrecy. "It created a protective blanket around the harasser and left the victim out in the cold," Speier said. Speier, who has worked on the issue since 2014, came forward in October with her own story of unwanted sexual contact from the chief of staff for the lawmaker she worked for as a congressional aide. "He kissed me and stuck his tongue in my mouth," said Speier, who has become a resource from women seeking advice on how to handle similar situations. "When it happened to me, it disgusted me. I kind of recoiled." Speier's proposals for reforms have attracted support from more than 100 members, including 19 Republicans. A group of conservative Republicans have championed a separate bill focused on banning the use of taxpayer dollars for settlements, and requiring past settlements to be disclosed and reimbursed. "What we do agree is that taxpayers should not be on the hook for misbehavior and for those settlements that are made," said Marsha Blackburn, a Republican representative who has advocated for the proposal. "We need to use that to make certain that workplaces are respectful," Blackburn said in an interview. A House committee is reviewing reforms with an eye to making recommendations in coming weeks. "I think that what we are doing is taking the best of all the ideas out there and putting them into one package," a senior House Republican aide said."""

# # # Parameters for chunking
# # sentences_per_chunk = 5
# # min_chunks = 3

# # # Split article into chunks
# # chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)

# # # Function to compute cosine similarity
# # def compute_cosine_similarities(article_probs, chunk_probs_list):
# #     similarities = []
# #     article_probs = np.array(article_probs).reshape(1, -1)  # Reshape for sklearn compatibility
# #     for chunk_probs in chunk_probs_list:
# #         chunk_probs = np.array(chunk_probs).reshape(1, -1)
# #         similarity = cosine_similarity(article_probs, chunk_probs)[0][0]
# #         similarities.append(similarity)
# #     return similarities

# # # Main execution
# # if len(chunks) == 0:
# #     print("No chunks generated. Ensure the article has sufficient sentences.")
# # else:
# #     print("Chunks:")
# #     for i, chunk in enumerate(chunks):
# #         print(f"Chunk {i+1}: {chunk}")

# #     # Compute topic probabilities for the overall article
# #     _, article_topic_probs = topic_model.transform([article_text])
# #     article_topic_probs = article_topic_probs[0]

# #     # Compute topic probabilities for each chunk
# #     chunk_topic_probs_list = []
# #     for i, chunk in enumerate(chunks):
# #         _, chunk_topic_probs = topic_model.transform([chunk])
# #         chunk_topic_probs_list.append(chunk_topic_probs[0])

# #     # Compute cosine similarities
# #     cosine_similarities = compute_cosine_similarities(article_topic_probs, chunk_topic_probs_list)
# #     print("\nCosine Similarities:")
# #     for i, sim in enumerate(cosine_similarities):
# #         print(f"Chunk {i+1} Cosine Similarity: {sim:.4f}")

# #     # Summary
# #     print("\nSummary:")
# #     print("Top 3 Topics for the Full Article:")
# #     full_article_topics = sorted(
# #         [(i, prob) for i, prob in enumerate(article_topic_probs)],
# #         key=lambda x: x[1],
# #         reverse=True
# #     )[:3]
# #     for rank, (topic, prob) in enumerate(full_article_topics):
# #         print(f"  Rank {rank+1}: Topic {topic}, Probability: {prob:.4f}")
# #     for chunk_index, chunk_probs in enumerate(chunk_topic_probs_list):
# #         chunk_top_topics = sorted(
# #             [(i, prob) for i, prob in enumerate(chunk_probs)],
# #             key=lambda x: x[1],
# #             reverse=True
# #         )[:3]
# #         print(f"\nTop 3 Topics for Chunk {chunk_index+1}:")
# #         for rank, (topic, prob) in enumerate(chunk_top_topics):
# #             print(f"  Rank {rank+1}: Topic {topic}, Probability: {prob:.4f}")

# #     # Plot cosine similarity for chunks
# #     plt.figure(figsize=(10, 5))
# #     plt.bar(range(1, len(cosine_similarities) + 1), cosine_similarities, color="skyblue")
# #     plt.xticks(range(1, len(cosine_similarities) + 1), [f"Chunk {i+1}" for i in range(len(chunks))])
# #     plt.ylabel("Cosine Similarity")
# #     plt.title("Cosine Similarity Between Full Article and Chunks")
# #     plt.show()





# ###PAIRWISE CHUNK SIMILARITY (instead of vs. full article)
# # import pandas as pd
# # from bertopic import BERTopic
# # from src.data.preprocessing import split_into_chunks
# # from src.utils.helper import compute_coherence_metrics
# # from scipy.spatial.distance import jensenshannon

# # # Load the pre-trained BERTopic model
# # topic_model = BERTopic.load("isot_topic_model")

# # # Define the article text
# # article_text = """Prominent Republican senators on Thursday embraced a push to overhaul rules for addressing sexual harassment in the U.S. Congress, signing on to a bill that would protect victims and require lawmakers to pay for their own settlements. The legislation builds on demands to lift the veil of secrecy around sexual harassment and misconduct on Capitol Hill, and has gained steam in recent months as a wave of women have come forward with accusations against prominent American men in politics, media and entertainment. The bipartisan push signaled momentum in the Republican-led U.S. Congress for overhauling a process for handling misconduct allegations that many lawmakers say is antiquated and stacked against victims. The Senate bill, called the Congressional Harassment Reform Act, draws from proposals that Senator Kirsten Gillibrand and Representative Jackie Speier, both Democrats, have been developing. "Congress is really behind the eight-ball. I think that, in many respects, the private sector has acted more swiftly than we have in terms of addressing sexual harassment," Speier said in an interview. High-profile Republican senators co-sponsoring the bill include John Cornyn, the Senate's No. 2 Republican; Ted Cruz; Joni Ernst and Lisa Murkowski. The legislation would require any member of Congress found liable for harassment to pay settlements themselves, rather than with taxpayer funds, as the current process allows. "Congress is not above the laws, and secret settlements with taxpayer money to cover up harassment should no longer be tolerated," Cruz said in a statement. Settlements would be made public automatically unless victims choose to keep them private. Outrage over sexual misconduct in politics helped to fuel an upset victory by Democrat Doug Jones in the U.S. Senate race in deeply conservative Alabama on Tuesday. Voters rejected the Republican candidate in the race, Roy Moore, who had been accused by multiple women of pursuing them when they were teenagers and he was in his 30s, including one woman who said he tried to initiate sexual contact with her when she was 14. Moore denied the allegations but many prominent Republicans distanced themselves from Moore, although President Donald Trump backed him. In Washington, allegations of sexual misconduct prompted the resignations last week of three lawmakers - Democratic Senator Al Franken, Democratic Representative John Conyers and Republican Representative Trent Franks. On Tuesday, Republican Representative Blake Farenthold said he would not seek re-election in November. Politico reported that the congressional Office of Compliance had paid $84,000 from a public fund on behalf of Farenthold to settle a sexual harassment claim in 2015. Reuters has been unable to verify the allegations against Farenthold, who has said that the charges were false and has denied wrongdoing. The 1995 law governing the process for complaints in Congress - created in the wake of a harassment scandal - has been criticized as ineffective. The lengthy and cumbersome process requires victims to go through mandatory mediation and requires complete secrecy. "It created a protective blanket around the harasser and left the victim out in the cold," Speier said. Speier, who has worked on the issue since 2014, came forward in October with her own story of unwanted sexual contact from the chief of staff for the lawmaker she worked for as a congressional aide. "He kissed me and stuck his tongue in my mouth," said Speier, who has become a resource from women seeking advice on how to handle similar situations. "When it happened to me, it disgusted me. I kind of recoiled." Speier's proposals for reforms have attracted support from more than 100 members, including 19 Republicans. A group of conservative Republicans have championed a separate bill focused on banning the use of taxpayer dollars for settlements, and requiring past settlements to be disclosed and reimbursed. "What we do agree is that taxpayers should not be on the hook for misbehavior and for those settlements that are made," said Marsha Blackburn, a Republican representative who has advocated for the proposal. "We need to use that to make certain that workplaces are respectful," Blackburn said in an interview. A House committee is reviewing reforms with an eye to making recommendations in coming weeks. "I think that what we are doing is taking the best of all the ideas out there and putting them into one package," a senior House Republican aide said."""

# # # Parameters for chunking
# # sentences_per_chunk = 5
# # min_chunks = 3

# # # Split article into chunks
# # chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)

# # # Check if there are enough chunks
# # if len(chunks) < 2:
# #     print("No chunks generated. Ensure the article has sufficient sentences.")
# # else:
# #     print("Chunks:")
# #     for i, chunk in enumerate(chunks):
# #         print(f"Chunk {i+1}: {chunk}")

# #     # Compute topic probabilities for each chunk
# #     chunk_probs = [topic_model.transform([chunk])[1][0] for chunk in chunks]

# #     # Compute pairwise JS values for the chunks
# #     js_values = []
# #     print("\nPairwise JS Values:")
# #     for i in range(len(chunk_probs)):
# #         for j in range(i + 1, len(chunk_probs)):
# #             js_value = jensenshannon(chunk_probs[i], chunk_probs[j])**2
# #             js_values.append(js_value)
# #             print(f"Chunk {i+1} vs Chunk {j+1}: JS Value = {js_value}")

# #     # Compute global coherence metrics
# #     metrics = compute_coherence_metrics(js_values)
# #     print("\nGlobal Coherence Metrics:")
# #     for key, value in metrics.items():
# #         print(f"{key}: {value}")
