competition = hubmap-kidney-segmentation

data/:
	mkdir -p $@
	kaggle competitions download -c $(competition) -p data
	unzip -qq data/$(competition).zip -d data/
	rm -rf data/$(competition).zip


.PHONY: dataset
