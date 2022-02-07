import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def image2zvals(img,  n_colors=64, n_training_pixels=800, rngs = 123): 
    # Image color quantization
    # img - np.ndarray of shape (m, n, 3) or (m, n, 4)
    # n_colors: int,  number of colors for color quantization
    # n_training_pixels: int, the number of image pixels to fit a KMeans instance to them
    # returns the array of z_values for the heatmap representation, and a plotly colorscale
   
    if img.ndim != 3:
        raise ValueError(f"Your image does not appear to  be a color image. It's shape is  {img.shape}")
    rows, cols, d = img.shape
    if d < 3:
        raise ValueError(f"A color image should have the shape (m, n, d), d=3 or 4. Your  d = {d}") 
        
    range0 = img[:, :, 0].max() - img[:, :, 0].min()
    if range0 > 1: #normalize the img values
        img = np.clip(img.astype(float)/255, 0, 1)
        
    observations = img[:, :, :3].reshape(rows*cols, 3)
    training_pixels = shuffle(observations, random_state=rngs)[:n_training_pixels]
    model = KMeans(n_clusters=n_colors, random_state=rngs).fit(training_pixels)
    
    codebook = model.cluster_centers_
    indices = model.predict(observations)
    z_vals = indices.astype(float) / (n_colors-1) #normalization (i.e. map indices to  [0,1])
    z_vals = z_vals.reshape(rows, cols)
    # define the Plotly colorscale with n_colors entries    
    scale = np.linspace(0, 1, n_colors)
    colors = (codebook*255).astype(np.uint8)
    pl_colorscale = [[sv, f'rgb{tuple(color)}'] for sv, color in zip(scale, colors)]
      
    # Reshape z_vals  to  img.shape[:2]
    return z_vals.reshape(rows, cols), pl_colorscale

def regular_tri(rows, cols):
    #define triangles for a np.meshgrid(np.linspace(a, b, cols), np.linspace(c,d, rows))
    triangles = []
    for i in range(rows-1):
        for j in range(cols-1):
            k = j+i*cols
            triangles.extend([[k,  k+cols, k+1+cols], [k, k+1+cols, k+1]])
    return np.array(triangles) 
       
def mesh_data(img, n_colors=32, n_training_pixels=800):
    rows, cols, _ = img.shape
    z_data, pl_colorscale = image2zvals(img, n_colors=n_colors, n_training_pixels=n_training_pixels)
    triangles = regular_tri(rows, cols) 
    I, J, K = triangles.T
    zc = z_data.flatten()[triangles] 
    tri_color_intensity = [zc[k][2] if k%2 else zc[k][1] for k in range(len(zc))]  
    return I, J, K, tri_color_intensity, pl_colorscale