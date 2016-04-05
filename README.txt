

Frameworks Used: 
	1) nltk 3.0 (http://www.nltk.org/)
	2) gutenberg parser (included in assignment)

	Used on Linux Mint Cinanamon 17.2 (Rafaela)

Part I and II usage (ngram.py):
	python ngram.py <[123,s] <train> <dev> <test>
	if an s is included, smoothing and interpolation will occur. Since the type of smoothing was not specified The type of smoothing used is a mixture of Jelinek-Mercer and add_one. If no smoothing is specified, unkown words will cause the perplexity to become infinite as was mentioend in class.

Part III:
	python languagemodel.py TRAINCORPUS.txt_TRAINCORPUS2.txt_TRAINCORPUSXXX.txt Holmes.lm_format.questions.txt 3s 

	The first parameter is a list of training corpuses seperated by an '_'. If the filename has an _ in it, rename it so that it can parse it correctly. You can train with 1...N files. The more N I train with, the better my N-grams perform.
	The 3s is recommended since other models perform poorly. 

	For testing:
		I used a greater split of the questions/answers on my dev file.
		In order to run evaluation you cannot use Holmes.lm.answers.txt you must use: my_answers.txt which is a greater split used on the dev file.

		To evaluate : 
			 cat output.txt | ./bestof5.pl > temp.txt
			./score.pl temp.txt my_answers.txt 						[MUST USE my_answers.txt]

	The training/testing files are not included as per the instructions. 