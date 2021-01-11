competition = hubmap-kidney-segmentation
logdir = $(TENSORBOARD_DIR)/$(message)

all: weights/fold0.pt \
	 weights/fold1.pt \
	 weights/fold2.pt \
	 weights/fold3.pt \
	 weights/fold4.pt

weights/fold%.pt: foldname = $(basename $(@F))
weights/fold%.pt: logfold = $(logdir)-$(foldname)
weights/fold%.pt: data/train/fold%.json
	python models/train.py --fin $< --fout $(@D)
	gsutil -m cp thresholds.png $(logfold)
	gsutil -m cp $(logfold)/train_end_params.pt $@

data/train/fold%.json: data/train/preprocessed
	python models/split.py --fin $^ --fout $(@D)


develop: data/train/preprocessed/
	python models/main.py --fin $^ --logdir=$(logdir)
	gsutil -m cp thresholds.png $(logdir)


infer:
	python models/infer.py


data/train/preprocessed/: data/train
	python models/preprocess.py --fin $^ --fout  $@


data/:
	mkdir -p $@
	kaggle competitions download -c $(competition) -p data
	unzip -qq data/$(competition).zip -d data/
	rm -rf data/$(competition).zip


push-data:
	mkdir -p .tmp_submit
	cp dataset-metadata.json .tmp_submit/
	cp requirements.txt .tmp_submit/
	cp setup.py .tmp_submit
	cp -R models .tmp_submit/models
	cp -R weights .tmp_submit/weights
	rm .tmp_submit/models/kernel-metadata.json
	kaggle datasets version -p .tmp_submit -r zip -m "$(message)"
	rm -rf .tmp_submit


push-kernels:
	kaggle kernels push -p models/


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


.PHONY: tolocal infer develop push-data push-kernels
.PRECIOUS: data/train/fold%.json
