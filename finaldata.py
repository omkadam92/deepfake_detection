import pandas as pd

df1 = pd.read_csv('combined_metadata.csv')
df2 = pd.read_csv('realfake_dataset.csv')

df1['path'] = 'deepfake_dataset/faces_224/'+df1['photoname']+'.jpg'
prefix = 'deepfake_dataset/real_vs_fake/real_vs_fake/'
df2['path'] = prefix + df2['path']

df = pd.concat([df1, df2], ignore_index=True)

df = df.drop(['original_width', 'original_height','label_no'], axis=1)
# If you need to save the changes back to CSV
df.to_csv('finaldata.csv', index=False)





