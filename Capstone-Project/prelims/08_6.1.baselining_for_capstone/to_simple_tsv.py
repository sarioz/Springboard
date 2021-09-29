import fileinput

tweet = "tweet"
sentiment = "sentiment"

for line in fileinput.input():
    line = line.strip()
    if len(line) <= 1:
        continue
    if line.startswith('# sent_enum = '):
        tweet = tweet.lstrip()
        print(tweet + "\t" + sentiment)
        sentiment = line.strip().split('\t')[-1]
        tweet = ""
    else:
        token = line.strip().split('\t')[0]
        tweet += " " + token

print(tweet.lstrip() + "\t" + sentiment)
