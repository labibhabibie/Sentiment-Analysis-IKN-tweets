import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Membaca dataset tweet
tweet_data = pd.read_csv("net_IKN_Nusantara.csv")

# Membaca dataset sentimen
sentiment_data = pd.read_csv("IKN_Nusantara_with_sentiment_labeled_lowercase_14b2d.csv")

# Membuat graph menggunakan networkx
G = nx.from_pandas_edgelist(tweet_data, 'from', 'to', create_using=nx.DiGraph())

# Menghitung setiap aktor yang berperan sebagai creator dan influencer
creators = [node for node, degree in G.in_degree() if degree > 0]
influencers = [node for node, degree in G.out_degree() if degree > 0]

# Menampilkan aktor-aktor yang berperan sebagai creator dan influencer
print("Creators:")
print(creators)
print("Influencers:")
print(influencers)

# Mengambil kolom sentiment_label dari sentimen_data berdasarkan aktor-aktor yang ditemukan
actor_sentiments = sentiment_data[sentiment_data['author'].isin(creators + influencers)][['author', 'sentiment_label']]

# Menampilkan hasil kolom sentiment_label sesuai dengan aktor-aktor
print("Sentiment Labels:")
print(actor_sentiments)

# Visualisasi jaringan sosial
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.2)
nx.draw_networkx(G, pos, with_labels=True, node_size=1000, node_color='lightblue')
plt.title("Social Network Analysis - IKN Nusantara")
plt.show()
