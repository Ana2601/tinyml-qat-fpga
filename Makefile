.PHONY: demo test clean

demo:
	python3 qat.py

test:
	python3 test_qat.py

clean:
	rm -rf data results __pycache__