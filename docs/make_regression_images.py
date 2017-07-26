
# coding: utf-8

# In[ ]:


from LOTUS_regression.predictors import load_data
import matplotlib.pyplot as plt

predictors = load_data('pred_baseline_pwlt.csv')

plt.figure(figsize=(8, 3))
plt.style.use('style.mplstyle')

plt.plot(predictors.index, predictors['qboA'])
plt.plot(predictors.index, predictors['qboB'])
plt.plot(predictors.index, predictors['enso'])
plt.plot(predictors.index, predictors['aod'])
plt.plot(predictors.index, predictors['solar'])


plt.legend(['QBO 1', 'QBO 2', 'ENSO', 'AOD', 'Solar'])


plt.savefig('source/images/predictors_default.png', dpi=400, bbox_inches='tight', transparent=True)


# In[ ]:


plt.figure(figsize=(8, 3))

plt.plot(predictors.index, predictors['linear_pre'])
plt.plot(predictors.index, predictors['linear_post'])

plt.legend(['Pre', 'Post'])

plt.savefig('source/images/predictors_pwlt.png', dpi=400, bbox_inches='tight', transparent=True)


# In[ ]:


plt.figure(figsize=(8, 3))

predictors = load_data('pred_baseline_ilt.csv')

plt.plot(predictors.index, predictors['linear_pre'])
plt.plot(predictors.index, predictors['linear_post'])

plt.plot(predictors.index, predictors['pre_const'])
plt.plot(predictors.index, predictors['post_const'])
plt.plot(predictors.index, predictors['gap_cons'])


plt.legend(['Pre', 'Post', 'Constant 1', 'Constant 2', 'Constant 3'])

plt.savefig('source/images/predictors_ilt.png', dpi=400, bbox_inches='tight', transparent=True)


# In[ ]:


plt.figure(figsize=(8, 3))

predictors = load_data('pred_baseline_eesc.csv')

plt.plot(predictors.index, predictors['eesc_1'])
plt.plot(predictors.index, predictors['eesc_2'])

plt.legend(['EESC 1', 'EESC 2'])

plt.savefig('source/images/predictors_eesc.png', dpi=400, bbox_inches='tight', transparent=True)

