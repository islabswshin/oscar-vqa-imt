import json


with open("pretrained_base/base/best/test-dev-vqa2_results.json", "r") as test_dev:
    test_dev_result = json.load(test_dev)

with open("pretrained_base/base/best/test-vqa2_results.json", "r") as test:
    test_result = json.load(test)

with open("pretrained_base/base/best/reproduce_result.json", "w") as reproduce_result:
	json.dump(test_dev_result+test_result, reproduce_result)