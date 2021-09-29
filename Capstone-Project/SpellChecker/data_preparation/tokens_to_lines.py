import fileinput

tweet = ""

for line in fileinput.input():
    line = line.strip()
    if len(line) <= 1:
        continue
    if line.startswith('# sent_enum = '):
        tweet = tweet.strip()
        if tweet: print(tweet)
        tweet = ""
    else:
        token = line.strip().split('\t')[0]
        tweet += " " + token

print(tweet)
