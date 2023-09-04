import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Membaca dataset tweet
tweet_data = pd.read_csv("net_IKN_Nusantara.csv")

# Membaca dataset sentimen
sentiment_data = pd.read_csv("IKN_Nusantara_with_sentiment_labeled_lowercase_14b2d.csv")

# Membuat graph menggunakan networkx
G = nx.from_pandas_edgelist(tweet_data, 'from', 'to', create_using=nx.DiGraph())

# Menghitung degree setiap node (aktor)
degree_centrality = nx.degree_centrality(G)

# Mengurutkan aktor berdasarkan degree mereka
sorted_creators = sorted([(node, degree_centrality[node]) for node in degree_centrality if degree_centrality[node] > 0], key=lambda x: x[1], reverse=True)
top_10_creators = sorted_creators[:10]

# Mengambil aktor influencer berdasarkan out_degree
out_degrees = dict(G.out_degree())
sorted_influencers = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
top_10_influencers = sorted_influencers[:10]

# # Mengambil kolom sentiment_label dari sentimen_data berdasarkan aktor-aktor yang ditemukan
# actor_sentiments = sentiment_data[sentiment_data['author'].isin([node[0] for node in top_10_creators + top_10_influencers])][['author', 'sentiment_label']]

# # Menampilkan aktor creator teratas beserta sentiment label
# print("Top 10 Creators:")
# for creator, degree in top_10_creators:
#     sentiment_label = actor_sentiments.loc[actor_sentiments['author'] == creator, 'sentiment_label'].values[0]
#     print(f"Creator: {creator} | Degree: {degree} | Sentiment Label: {sentiment_label}")

# # Menampilkan aktor influencer teratas beserta sentiment label
# print("\nTop 10 Influencers:")
# for influencer, degree in top_10_influencers:
#     sentiment_label = actor_sentiments.loc[actor_sentiments['author'] == influencer, 'sentiment_label'].values[0]
#     print(f"Influencer: {influencer} | Degree: {degree} | Sentiment Label: {sentiment_label}")

# Mengambil kolom sentiment_label dari sentimen_data berdasarkan aktor-aktor yang ditemukan
actor_sentiments = sentiment_data[sentiment_data['author'].str.lower().isin([node[0].lower() for node in top_10_creators + top_10_influencers])][['author', 'sentiment_label']]

# Menampilkan aktor creator teratas beserta sentiment label
print("Top 10 Creators:")
for creator, degree in top_10_creators:
    sentiment_label = actor_sentiments.loc[actor_sentiments['author'].str.lower() == creator.lower(), 'sentiment_label'].values[0]
    print(f"Creator: {creator} | Degree: {degree} | Sentiment Label: {sentiment_label}")

# Menampilkan aktor influencer teratas beserta sentiment label
print("\nTop 10 Influencers:")
for influencer, degree in top_10_influencers:
    sentiment_label = actor_sentiments.loc[actor_sentiments['author'].str.lower() == influencer.lower(), 'sentiment_label'].values[0]
    print(f"Influencer: {influencer} | Degree: {degree} | Sentiment Label: {sentiment_label}")


# Visualisasi jaringan sosial
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.2)
nx.draw_networkx(G, pos, with_labels=True, node_size=1000, node_color='lightblue')
plt.title("Social Network Analysis - IKN Nusantara")
plt.show()
