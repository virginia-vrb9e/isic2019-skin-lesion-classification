default:
	@cat makefile

run425:
	mkdir -p logs
	rm -f run.log
	rm -f wandb/latest-run/*.*
	rm -f /home/ubuntu/dpy8wq/isic2019/checkpoints/*.*
	rm -f *.csv
	RUN_ID=$$(date +%Y%m%d_%H%M%S); \
	LOG="logs/train_$${RUN_ID}.log"; \
	ZIP="logs/run_bundle_$${RUN_ID}.zip"; \
	PYTHONUNBUFFERED=1 python 4_25_EFN_HS_Ablation1.py --wandb 2>&1 | tee -a "$$LOG"; \
	zip -r "$$ZIP" "$$LOG" run.log wandb /home/ubuntu/dpy8wq/isic2019/checkpoints *.csv 4_25_EFN_HS_Ablation1.py

run428:
	mkdir -p logs
	rm -f run.log
	rm -f wandb/latest-run/*.*
	rm -f /home/ubuntu/dpy8wq/isic2019/checkpoints/*.*
	rm -f *.csv
	RUN_ID=$$(date +%Y%m%d_%H%M%S); \
	LOG="logs/train_$${RUN_ID}.log"; \
	ZIP="logs/run_bundle_$${RUN_ID}.zip"; \
	PYTHONUNBUFFERED=1 python 4_28_EFN_HS_Ablation1.py --wandb 2>&1 | tee -a "$$LOG"; \
	zip -r "$$ZIP" "$$LOG" run.log wandb /home/ubuntu/dpy8wq/isic2019/checkpoints 4_28_EFN_HS_Ablation1.py; \
	find . -maxdepth 1 -type f -name "*.csv" -exec zip "$$ZIP" {} \;
