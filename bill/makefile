default:
	@cat makefile

convert:
	rm -f mbv3-small.py
	jupyter nbconvert --to script mbv3-small.ipynb

run:
	mkdir -p logs
	rm -f *.png
	rm -f run.log
	rm -f wandb/latest-run/*.*
	rm -f weights/*
	rm -f /home/ubuntu/dpy8wq/isic2019/checkpoints/*.*
	rm -f *.csv
	RUN_ID=$$(date +%Y%m%d_%H%M%S); \
	LOG="logs/train_$${RUN_ID}.log"; \
	ZIP="logs/run_bundle_$${RUN_ID}.zip"; \
	PYTHONUNBUFFERED=1 python mbv3-small.py --wandb 2>&1 | tee -a "$$LOG"; \
	zip -r "$$ZIP" run.log wandb weights /home/ubuntu/dpy8wq/isic2019/checkpoints \
	*.csv mbv3-small.py *.png "$$LOG"
