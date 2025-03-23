Yiwei recommended: 

1. Good Idea, can try to publish after the course, do something on 3 million datasets. For the course project, training can take long (suppose we have 4 collab, each a A100 GPU), we can do it on 30 thousand dataset, smaller scale
2. Text generation part: Yiwei suggested that instead of running local models, we can **buy GPT 4-o API token**: 3 million tokens about 100 dollars; we don't have to do that much, 10 dollars is enough. We can do it by batch inference,  saving even more.
3. We are concerned about illed classification, but dont worry. Yiwei suggest that, we can instead, do the 80 keywords on the caption, also by GPT.
4. Diffusion part: mask 遮掉后有残留部分是很正常的，但是仍然 descent, 因为有残余的部分, 人仍然会识别出"这里没有人/狗". 不需要再精进 mask. 在生成结束后, can do quality check, 剔除掉质量有一定问题的 images (check papers on image generation quality check)
5. loss function part: don't necessaily have to change the loss function (bu we are encouraged to look into it); can put similar things, espectially the newly generated ones and the original one, in one batch.
6. spilit 30000 pictures into training set and do cross validation. show our fine-tuned performance is better



单张 A100

NLP: 20FPS

Yolo: 20FPs 

Diffusion，2~4 FPS （多张 15 FPS, 4 张 e.g)



一个流程: 0.05+ 0.05 + 0.33 =0.43 秒



86,400 秒

一天: 可以搞完 



所以假设一人用一张 A100 跑一天：86,400 * 2 =172,800 张

四人各用一张 A100 跑一天：172800 * 4 =691,200 张 （上限



提取关键词+ 分类：API





总结：

1. 买 API，不多，1-5 刀就可以。并且不需要一个图片&文字用一个 token. 可以做 batch inference.
2. 30000 张不用计算资源，colab 就可以（yiwei 也推荐说 colab 挺不错的
3. diffusion 结束后可以做 quality check
4. ddd
5. 
