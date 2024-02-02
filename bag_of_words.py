from sklearn.feature_extraction.text import CountVectorizer

text_data = ["Графічні Lorem та обычные обычные обычные Lorem оператори це добре знають, насправді всі професії, що займаються всесвітом спілкування, мають стабільний зв'язок із цими словами, але що це таке? ", "Це послідовність латинських слів , які, як вони розташовані , не формуйте речень із повним змістом, а дайте життя тестовому тексту, корисному заповнити пробіли", "За допомогою інструменту Lorem Ipzum , ви можете вставляти тексти безпосередньо з ключовими словами, які слугуватимуть для індексування вашого веб -сайту. "]

bow = CountVectorizer(stop_words='english')

#fir the data
bow.fit(text_data)

# get the vocabulary list
#print(bow.get_feature_names_out())

bow_features = bow.fit_transform(text_data)
#print(bow_features)

bow_features_array = bow_features.toarray()
#print(bow_features_array)

print(bow.get_feature_names_out())
for sentence, feature in zip(text_data, bow_features_array):
    print(sentence)
    print(feature)
