import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Membaca dataset tweet
tweet_data = pd.read_csv("net_IKN_Nusantara.csv")

# Membaca dataset sentimen
sentiment_data = pd.read_csv("IKN_Nusantara_with_sentiment_labeled_lowercase_14b2d_new.csv")

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

# Mengambil kolom sentiment_label dari sentimen_data berdasarkan aktor-aktor yang ditemukan
actor_sentiments = sentiment_data[sentiment_data['author'].str.lower().isin([node[0].lower() for node in top_10_creators + top_10_influencers])][['author', 'sentiment_label']]

# Menampilkan aktor creator teratas beserta sentiment label
creator_results = []
for creator, degree in top_10_creators:
    sentiment_label = actor_sentiments.loc[actor_sentiments['author'].str.lower() == creator.lower(), 'sentiment_label'].values[0]
    creator_results.append({'Creator': creator, 'Degree': degree, 'Sentiment Label': sentiment_label})

# Menampilkan aktor influencer teratas beserta sentiment label
influencer_results = []
for influencer, degree in top_10_influencers:
    sentiment_label = actor_sentiments.loc[actor_sentiments['author'].str.lower() == influencer.lower(), 'sentiment_label'].values[0]
    influencer_results.append({'Influencer': influencer, 'Degree': degree, 'Sentiment Label': sentiment_label})

# Menyimpan hasil dalam satu file Excel dengan sheet yang berbeda
with pd.ExcelWriter("social_analysis_results_new.xlsx") as writer:
    creator_results_df = pd.DataFrame(creator_results)
    creator_results_df.to_excel(writer, sheet_name='Top Creators', index=False)

    influencer_results_df = pd.DataFrame(influencer_results)
    influencer_results_df.to_excel(writer, sheet_name='Top Influencers', index=False)
