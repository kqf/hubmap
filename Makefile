competition = data-science-bowl-2018

data/:
	mkdir -p $@
	kaggle competitions download -c $(competition) -p data
	unzip -qq data/$(competition).zip -d data/


.PHONY: dataset
