import fileinput

from google.cloud import language_v1


client = language_v1.LanguageServiceClient()

line_num = 0

print("Tweet\tCurated sentiment\tG_Sentiment_Score\tG_Sentiment_Magnitude")

# we read input from stdin
for line in fileinput.input():
	line_num += 1
	if line_num == 1:
		continue
	# # so we don't break the bank while debugging
	# if line_num == 5:
	#	break
	tweet, provided_sentiment = line.strip().split('\t')
	document = language_v1.Document(content=tweet, type_=language_v1.Document.Type.PLAIN_TEXT)
	# We don't specify language which lets Google figure it out
	try:
		detected_sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
		print("{}\t{}\t{:.2f}\t{:2f}".format(tweet, provided_sentiment, detected_sentiment.score, detected_sentiment.magnitude))
	except:
		print("{}\t{}\tException\tException".format(tweet, provided_sentiment))

