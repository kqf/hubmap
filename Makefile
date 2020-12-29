competition = hubmap-kidney-segmentation


infer:
	python models/infer.py


train: data/train/preprocessed
	python models/main.py --fin $^


data/train/preprocessed/: data/train
	python models/preprocess.py --fin $^ --fout  $@


data/:
	mkdir -p $@
	kaggle competitions download -c $(competition) -p data
	unzip -qq data/$(competition).zip -d data/
	rm -rf data/$(competition).zip


tolocal:
	mkdir -p data/train

	scp $(instance):~/hubmap/data/HuBMAP-20-dataset_information.csv data/
	scp $(instance):~/hubmap/data/train.csv data/

	scp $(instance):~/hubmap/data/train/2f6ecfcdf.tiff data/train
	scp $(instance):~/hubmap/data/train/2f6ecfcdf.json data/train
	scp $(instance):~/hubmap/data/train/2f6ecfcdf-anatomical-structure.json data/train

	mkdir -p hubmap/data/test
	scp $(instance):~/hubmap/data/test/b2dc8411c.tiff data/test
	scp $(instance):~/hubmap/data/test/b2dc8411c-anatomical-structure.json data/test


.PHONY: tolocal
