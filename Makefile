APPLICATION_MODULE=text_generator
TEST_MODULE=tests

PYTHON3_DIR=/usr/local/bin/python3 ## this path is available on linux and mac

# @see http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.DEFAULT_GOAL := help
.PHONY: help
help: ## provides cli help for this makefile (default)
	@grep -E '^[a-zA-Z_0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: dist
dist:
	. venv/bin/activate; python setup.py sdist

.PHONY: tests
tests: ## run automatic tests
	. venv/bin/activate; pytest $(TEST_MODULE)/

.PHONY: clean
clean :
	rm -rf dist
	rm -rf venv
	rm -f .coverage
	rm -rf *.egg-info

.PHONY: freeze
freeze: ## freeze the dependencies version in requirements.txt
	. venv/bin/activate; pip freeze | grep -v "$(APPLICATION_MODULE)" > requirements.txt

.PHONY: update_requirements
update_requirements: ## update the project dependencies based on setup.py declaration
	rm -rf venv
	make venv
	. venv/bin/activate; pip install .
	$(MAKE) freeze

.PHONY: install_requirements_dev
install_requirements_dev: venv ## install pip requirements for development
	. venv/bin/activate; pip install -r requirements_dev.txt

.PHONY: install_requirements
install_requirements: ## install pip requirements based on requirements.txt
	. venv/bin/activate; pip install -r requirements.txt

.PHONY: venv
venv: ## build a virtual env for python 3 in ./venv
	virtualenv venv -p $(PYTHON3_DIR)
	@echo "\"source venv/bin/activate\" to activate the virtual env"