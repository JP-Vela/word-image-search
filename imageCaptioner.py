import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from imageEnumerator import Enumerator
import pickle

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("Model loaded")

full_path = "/home/jp/Projects/word-image-search/images"
enumerator = Enumerator(full_path)
image_paths = enumerator.get_image_paths()


def get_docs():
    image_descs = []
    image_ids = []
    image_meta = []
    id = 0

    cached_data = []
    with open('imageCache.pickle', 'rb') as handle:
             cached_data = pickle.load(handle)



    for path in image_paths:
        done = False
        for i in range(len(cached_data['meta'])):

            if cached_data['meta'][i]['path'] == path:
                image_descs.append(cached_data['docs'][i])
                image_meta.append(cached_data['meta'][i])
                image_ids.append(str(id))
                done = True
            continue

        if done:
            id+=1
            continue

        raw_image = Image.open(full_path+"/"+path).convert('RGB')

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")

        out = model.generate(**inputs)
        desc = processor.decode(out[0], skip_special_tokens=True)

        image_descs.append(desc)
        image_ids.append(str(id))
        image_meta.append({'path':path})
        id+=1
        print(f'Image {id} processed')

    documents = {'docs':image_descs, 'meta': image_meta, 'ids': image_ids}
    with open('imageCache.pickle', 'wb') as handle:
        pickle.dump(documents, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return documents