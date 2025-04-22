for i, (img_path, img_ID, caption, caption_id) in enumerate(image_caption_pairs):



我们想要把这个 batch 的 caption 和 image 整合成这样:

```python
captions   = 
{1: "A person is cutting a roast with a fork and knife.",
 2: "Dining room table set for a casual meal,with flowers.",
 3: "A large bus and some people on the street.",
 4: "A bicycle parked in a kitchen with a stove and cabinets.",
 5: "A group of men at a table preparing food together.",
 6: "Adults using laptop computers while sitting at outer venue",
 7: "A man preparing desserts in a kitchen covered in frosting",
 8: "Two people in a food truck, one looking at an order"
}

true_images = 
{1: "train2014/COCO_train2014_000000161919.jpg", 
 2: "train2014/COCO_train2014_000000071631.jpg",
 3: "train2014/COCO_train2014_000000392136.jpg", 
 4: "train2014/COCO_train2014_000000398494.jpg",
 5: "train2014/COCO_train2014_000000405613.jpg", 
 6: "train2014/COCO_train2014_000000170558.jpg",
 7: "train2014/COCO_train2014_000000384029.jpg", 
 8: "train2014/COCO_train2014_000000090570.jpg"}

neg_images  = 
{
  1: ["DualImageFolder/161919-person.jpg"],
  2: ["DualImageFolder/71631-vase.jpg", "DualImageFolder/71631-dining table.jpg"],
  3: ["DualImageFolder/392136-person.jpg","DualImageFolder/392136-bus.jpg"],
  4: ["DualImageFolder/398494-bicycle.jpg"],
  5: ["DualImageFolder/405613-person.jpg", "DualImageFolder/405613-dining table.jpg"],
  6: ["DualImageFolder/170558-laptop.jpg", "DualImageFolder/170558-person.jpg"],
  7: ["DualImageFolder/384029-cake.jpg", "DualImageFolder/384029-person.jpg"],
  8: ["DualImageFolder/90570-person.jpg"]
}
```

其中, true images 对于每个 batch index 都存在, 直接饮用 image_path 即可.

negative images 存在 DualImageFolder 下, 其格式固定为 

```
{img_id}-{...}.jpg
```

请你把数据变成这样.





OK! 还有最后一个:

for each i, `dual_caption_data[i]["rewrites"]` 是这样的：

```python
dual_caption_data[1]["rewrites"] = 
[{'removed_noun': 'man', 'removed_noun_category': 'person', 'rewritten_caption': '"Desserts in a kitchen covered in frosting."'}, 
 {'removed_noun': 'desserts', 'removed_noun_category': 'cake', 'rewritten_caption': '"A man in a kitchen covered in frosting."'}]
```

请你把它对应上 neg_images, 做成对照的格式