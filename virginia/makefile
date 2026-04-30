default:
	@cat makefile

test:
	python rn50_base_va.py --num_epochs 1 --wandb


run_mini:
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
	PYTHONUNBUFFERED=1 python \
	rn50_base_va_mini.py \
	--num_epochs 1 --wandb 2>&1 | tee -a "$$LOG"; \
	zip -r "$$ZIP" "$$LOG" run.log wandb weights /home/ubuntu/dpy8wq/isic2019/checkpoints \
	*.csv \
	rn50_base_va_mini.py \
	*.png


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
	PYTHONUNBUFFERED=1 python rn50_base_va.py --num_epochs 60 --wandb 2>&1 | tee -a "$$LOG"; \
	zip -r "$$ZIP" "$$LOG" run.log wandb weights /home/ubuntu/dpy8wq/isic2019/checkpoints \
	*.csv rn50_base_va.py *.png

rezip:
	RUN_ID='20260430_015233'; \
	LOG="logs/train_$${RUN_ID}.log"; \
	ZIP="logs/run_bundle_$${RUN_ID}.zip"; \
	zip -r "$$ZIP" "$$LOG" run.log wandb weights /home/ubuntu/dpy8wq/isic2019/checkpoints \
	*.csv rn50_base_va.py *.png

