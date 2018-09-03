import nltk
import os
import random
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk import NaiveBayesClassifier,  classify

def init_lists(folder):
#Menginisialisasi data
	a_list = []
	file_list = os.listdir(folder)
	for a_file in file_list:
		f = open(folder + a_file, 'r')
		a_list.append(f.read())
	f.close()
	return a_list

def preprocess(sentence):
#Preprocessing Data
	lemmatizer = WordNetLemmatizer()
	words =[]
	for word in word_tokenize(sentence.decode('utf-8','ignore')):
		if isWord(word):
			words.append(word)

	for word in words:
		lemmatizer.lemmatize(word.lower())

	return words

def isWord(word):
	return(word != 'forwarded' and word != 'subject' and word[0] != '/' and not isNumeric(word) and len(word) > 2)

def isNumeric(word):
	for c in word:
		if(c == '1' or c == '2' or c == '3' or c == '4' or c == '5' or c == '6' or c == '7' or c == '8' or c == '9' or c == '0'):
			return True
	return False

def get_features_bow(text):
#Mengambil fitur-fitur data
	stoplist = stopwords.words('english')
	return{word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}

def get_features(text):
	stoplist = stopwords.words('english')
	return{word: True for word in preprocess(text) if not word in stoplist}

def train(train_set):
#Melatih classifier
	print('Training Classifier')
	classifier = NaiveBayesClassifier.train(train_set)
	return classifier

def evaluate(train_set, test_set, classifier):
	classifier.show_most_informative_features(20)
	#print('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
	#print('Accuracy on the test set = ' + str(classify.accuracy(classifier, test_set)))

def init_data(folder):
	print('Inisialisasi Data')
	spam = init_lists(folder + '/spam/')
	ham = init_lists(folder + '/ham/')
	all_emails = [(email, 'spam') for email in spam]
	all_emails += [(email, 'ham') for email in ham]
	random.shuffle(all_emails)

	print('Ekstraksi Fitur')
	all_features = [(get_features_bow(email),label) for (email, label) in all_emails]

	return all_features

def read_file(file_name):
	file_name = raw_input('nama file : ')
	email = open('test/' + file_name).read()
	feature = get_features_bow(email)
	return classifier.classify(feature)

def read_folder(folder_name):
	spam_count = 0
	ham_count = 0
	file_list = os.listdir(folder_name)
	for file in file_list:
		email = open(folder_name + file, 'r').read()
		feature = get_features_bow(email)
		if(classifier.classify(feature) == 'spam'):
			print('SPAM : ' + file)
			spam_count = spam_count + 1
		else:
			print('HAM  : ' + file)
			ham_count = ham_count + 1
	return spam_count, ham_count

def get_confusion_matrix(folder_name):
	TP = 0
	TN = 0
	FP = 0
	FN = 0

	file_list = os.listdir(folder_name + '/spam/')
	for file in file_list:
		email = open(folder_name + '/spam/' + file, 'r').read()
		feature = get_features_bow(email)
		if(classifier.classify(feature) == 'spam'):
			TP = TP + 1
		elif(classifier.classify(feature) == 'ham'):
			FN = FN + 1

	file_list = os.listdir(folder_name + '/ham/')
	for file in file_list:
		email = open(folder_name + '/ham/' + file, 'r').read()
		feature = get_features_bow(email)
		if(classifier.classify(feature) == 'ham'):
			TN = TN + 1
		elif(classifier.classify(feature) == 'spam'):
			FP = FP + 1
	return TP, TN, FP, FN

def main():
	#folder = raw_input("Training Set : ")
	#train_set = init_data('dataset/' + folder)
	#classifier = train(train_set)

	folder = raw_input("Folder Test Set : ")
	#test_set = init_data('dataset/' + folder)
	TP, TN, FP, FN = get_confusion_matrix('dataset/' + folder + '/')

	print ('True Spam   : ' + str(TP))
	print ('True Ham    : ' + str(TN))
	print ('False Spam  : ' + str(FP))
	print ('False Ham   : ' + str(FN))
	#evaluate(train_set, test_set, classifier)

if __name__ == '__main__':
	main()
