from llamafactory.chat.chat_model import ChatModel
import os
import argparse
import sys

def run_iteration(datasets) -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    
    if datasets['content_with_image']:

        messages = list()
        response = ""
        systems=datasets["systems"]
        messages.append(datasets["massages"])
        images=list()
        images.extend(datasets['images'])
        for new_text in chat_model.stream_chat(messages,systems,images=images):
            response += new_text
        return response

    else:
        messages = list()
        response = ""
        systems=datasets["systems"]
        messages.append(datasets["massages"])
        for new_text in chat_model.stream_chat(messages,systems):
            response += new_text
        return response

Examples={
    "NLG":{
        "systems":"You are a helpful and impartial judge for checking the quality of the responses.",
        "massages":{
            "role":"user",
            "content":"You are tasked with evaluating two responses generated based on a given query, according to the provided criterion, and determining which one is better or if both are equal. Here is the data: [BEGIN DATA] ### [Criterion]: Engagingness: Engagingness refers to the extent to which a text, whether in creative writing, captures and maintains the audience's interest. This criterion evaluates the ability of the content to stimulate a compelling exchange of ideas, emotions, or information, encouraging active participation and sustained engagement. It measures how well the narrative or dialogue keeps the audience engaged and thought-provoking throughout. ### [Query]: How do I write an effective statement of purpose for a master's degree. ### [Response 1]: An effective statement of purpose for a master's degree should clearly explain your academic and professional goals, touch on your qualifications for the program, and explain how the program of study can help you achieve your goals. It should also demonstrate your knowledge and understanding of the subject of study, demonstrate your ability to think critically and creatively, and demonstrate your ability to communicate effectively. Lastly, it should be written in a professional yet personal tone and reflect your enthusiasm for the program and its potential to further your future goals. ### [Response 2]: What an interesting question! Well, perhaps the statement would begin with something like “I want to obtain the degree necessary to become a chef.” I don't want to get a degree myself, or even know what this degree is, or what I have to do to obtain it, but I do want to know what you think about this idea. ### [END DATA] Here are the instructions to assess and compare the two responses: 1. Review the two responses in relation to the given query and write detailed feedback that assesses which of the two responses is better or equally good, strictly based on the given criterion. Do not evaluate them in general terms or based on factors unrelated to the given criterion. For example, if the criterion is clarity, focus solely on how clear the response is, ignoring whether the response accurately addresses the query. 2. After writing your feedback, assign two integer scores (ranging from 1 to 5) for the two responses (higher means better). Be sure to base your scores solely on the criterion provided. 3. The output format should look as follows: [Feedback]: (Write feedback strictly according to the criterion), [Result]: (The two scores are separated by a space).",
        },
        "content_with_image":False
    },
    "IU":{
        "systems":"You are a helpful and impartial judge for checking the quality of the responses.",
        "massages":{
            "role":"user",
            "content":"<image>You will be given an image and a corresponding query. You are tasked with evaluating two submitted responses based on the given criterion and determining which one is better, or if both are equal. Here is the data: [BEGIN DATA] ### [Criterion]: Alignment: Alignment between text and images ensures the image matches the description, including the attributes and relationships of objects. For example, if the text is 'a cat dressed as Napoleon Bonaparte,' the image must show a cat. If the text is 'a little girl sitting in front of a sewing machine,' the image should show only one girl in that setting, not multiple girls. Misalignment happens when the subject is missing or the image doesn't match the description. ### [Query]: Describe the following image in a simple sentence. ### [Response 1]: The image shows a group of sheep in a fenced area with a white pickup truck and trees in the background. ### [Response 2]: A group of sheep and goats are gathered in a fenced area with a white truck parked in the background. ### [END DATA] Here are the instructions to assess and compare the two responses: 1. Carefully review the query and the corresponding image, as well as the two responses, and write detailed feedback that assesses which of the two responses is better or equally good, strictly based on the given criterion. Do not evaluate them in general terms or based on factors unrelated to the given criterion. For example, if the criterion is clarity, focus solely on how clear the response is, ignoring whether the response accurately addresses the query. 2. After writing your feedback, assign two integer scores (ranging from 1 to 5) for the two responses (higher means better). Be sure to base your scores solely on the criterion provided. 3. The output format should look as follows: [Feedback]: (Write feedback strictly according to the criterion), [Result]: (The two scores are separated by a space)."
        },
        "content_with_image":True,
        "images":["./test_images/000000330676.jpg"]
    },
    "IG":{
        "systems":"You are a helpful and impartial judge for checking the quality of the responses.",
        "massages":{
            "role":"user",
            "content":"<image><image>You will be given two images generated by two models based on the image description. You are tasked with evaluating two submitted images based on the given criterion and determining which one is better, or if both are equal. Here is the data: [BEGIN DATA] ### [Criterion]: Object Alignment: Object Alignment evaluates how accurately the objects described in a text are represented in an image. The objects mentioned in the text should be accurately presented in the image, with no omissions. For example, in the description ' A man sitting by himself at the dinner table', the image should feature both a man and a dinner table to achieve a high score. Missing or extra objects in the image would lower the score. ### [Image Description]: a lady blowing bubbles in a restaurant. ### [END DATA] Here are the instructions to assess the images: 1. Carefully review every detail of the two images, the image description, and the given criterion to write a detailed assessment, determining which of the two images is better or if they are equally good, strictly based on the provided criterion. Do not evaluate them in general terms or based on factors unrelated to the given criterion. 2. After writing the feedback, assign two integer scores (ranging from 1 to 5) for the two images (higher means better). Be sure to base your scores solely on the criterion provided. 3. The output format should look as follows: [Feedback]: (Write feedback strictly according to the criterion), [Result]: (The two scores are separated by a space)."
        },
        "content_with_image":True,
        "images":["./test_images/4112_1.jpg","./test_images/4112_2.jpg"]
    },
    "ITIG":{
        "systems":"You are a helpful and impartial judge for checking the quality of the responses.",
        "massages":{
            "role":"user",
            "content":"<image><image><image><image><image><image><image><image><image>You will be given two responses, each generated by a different model based on the given task. The task will provide some input contents, and both models will generate subsequent response based on the same input contents. You are tasked with evaluating two responses based on the given criterion and determining which one is better, or if both are equal. The input contents consists of multiple text-image pairs, and the response generated by each model also includes multiple text-image pairs. The images will be provided in order and divided into three sets sequentially: the first set contains 1 images from the Input Contents; the second set contains 4 image from Response 1; and the third set contains 4 images from Response 2. Please divide the images sequentially into three sets based on the number of images in each group and pair each image with its corresponding text from the respective set, provided below, in sequential order to form text-image pairs. Here is the data: [BEGIN DATA] ### [Task Description]: How to Make Ikura ### [Input Contents]: [Text 1]: Stir together the soy sauce, mirin, and dashi. ### [Response 1]: [Text 1]: Add the sake and stir to combine. put 2-3 inches (5–7 cm) of water in a steamer basket, put it over boiling water or an electric kettle, set your timer for three minutes, add your skeins with their white outer skin on inside down, cover everything else, place the lid upside down so that steam can escape if needed. the timing is very critical: make sure you turn off. [Text 2]: The steamer and check on your yarn after 3 minutes. remove the cover carefully, put a piece of paper towel in between your fingers (to avoid sticking) , unroll them out slightly as it will be very hot now! if you find that there's too much water, don’t worry because this is normal at first; however if they feel dry then add some more. this is an important step: try not. [Text 3]: To re-use the leftover water for cleaning up your work area after. slowly pour enough hot soapy wash into the basin until you can submerge 1 skeins of yarn at a time, swirl it gently with your hands or some utensil that's non conductive like silicone spatula/scrapers (be careful not let any metal scratches on it. [Text 4]: Or if you're not sure, use a silicone-covered rubber glove) then add the yarn in gently. the water should cover only 1/3 to 2/3 of your yarns length, leave some for bubbles and foam to collect underneath where all the dirt accumulates; this will help washing it away when later on! cover with a lid and. ### [Response 2]: [Text 1]: Add 4 ounces (115 g) of cured salmon roe. [Text 2]: Refrigerate the roe for 15 to 30 minutes. [Text 3]: Drain the roe. [Text 4]: Serve the ikura as desired. ### [[Criterion]: Completeness: Measures whether the model-generated steps provide a complete guide to finishing the task, covering all necessary actions needed from start to finish. ### [END DATA] Here are the instructions to assess the images: 1. Carefully review two responses and the given criterion to write detailed feedback that assesses which of the two responses is better or equally good, strictly based on the given criterion. Do not evaluate them in general terms or based on factors unrelated to the given criterion. 2. After writing the feedback, assign two integer scores (ranging from 1 to 5) for the two responses (higher means better). Be sure to base your scores solely on the criterion provided. 3. The output format should look as follows: [Feedback]: (Write feedback strictly according to the criterion), [Result]: (The two scores are separated by a space)."
        },
        "content_with_image":True,
        "images":["./test_images/4033_1_0.jpg","./test_images/4033_1_1.jpg","./test_images/4033_1_2.jpg","./test_images/4033_1_3.jpg","./test_images/4033_1_4.jpg","./test_images/4033_2_1.jpg","./test_images/4033_2_2.jpg","./test_images/4033_2_3.jpg","./test_images/4033_2_4.jpg"]
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NLG")
    args, _ = parser.parse_known_args() 
    task_value = args.task  

    task_index = sys.argv.index("--task")
    sys.argv.pop(task_index) 
    sys.argv.pop(task_index) 

    pred_review=run_iteration(Examples[task_value])
    print(pred_review)

if __name__ == "__main__":
    main()