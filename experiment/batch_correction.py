import logging
from scib.integration import *
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def combat_func(adata, batch_key):
    """Combat correction"""
    logger.info("Starting ComBat batch correction")
    try:
        adata_corrected = combat(adata, batch_key)
        logger.info("ComBat correction completed")
        return adata_corrected
    except Exception as e:
        logger.error(f"An error occurred in ComBat: {e}")
        raise

def harmony_func(adata, batch_key):
    """Harmony batch correction"""
    logger.info("Starting Harmony batch correction")
    try:
        adata_corrected = harmony(adata, batch_key)
        logger.info("Harmony correction completed")
        return adata_corrected
    except Exception as e:
        logger.error(f"An error occurred in Harmony: {e}")
        raise

def mnn_func(adata, batch_key):
    """MNN batch correction"""
    logger.info("Starting MNN batch correction")
    try:
        adata_corrected = mnn(adata, batch_key)
        logger.info("MNN correction completed")
        return adata_corrected
    except Exception as e:
        logger.error(f"An error occurred in MNN: {e}")
        raise

def scanorama_func(adata, batch_key):
    """Scanorama batch correction"""
    logger.info("Starting Scanorama batch correction")
    try:
        adata_corrected = scanorama(adata, batch_key)
        logger.info("Scanorama correction completed")
        return adata_corrected
    except Exception as e:
        logger.error(f"An error occurred in Scanorama: {e}")
        raise

def scvi_func(adata, batch_key):
    """scVI batch correction"""
    logger.info("Starting scVI batch correction")
    try:
        adata_corrected = scvi(adata, batch_key)
        logger.info("scVI correction completed")
        return adata_corrected
    except Exception as e:
        logger.error(f"An error occurred in scVI: {e}")
        raise

def desc_func(adata, batch_key):
    """DESC batch correction"""
    logger.info("Starting DESC batch correction")
    try:
        adata_corrected = desc(adata, batch_key)
        logger.info("DESC correction completed")
        return adata_corrected
    except Exception as e:
        logger.error(f"An error occurred in DESC: {e}")
        raise

def sphering_func(adata):
    """Sphering data transformation"""
    logger.info("Starting Sphering transformation")
    try:
        # Store the original obs and var
        original_obs = adata.obs.copy()
        original_var = adata.var.copy()

        # Get the data matrix
        X = adata.X.copy()

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute the covariance matrix
        cov = np.cov(X_scaled, rowvar=False)

        # Compute the inverse square root of the covariance matrix
        cov_inv_sqrt = np.linalg.inv(sqrtm(cov))

        # Apply the whitening transformation
        X_whitened = np.dot(X_scaled, cov_inv_sqrt)

        # Create a new AnnData object with the whitened data
        adata_transformed = adata.copy()
        adata_transformed.X = X_whitened

        # Restore the original obs and var
        adata_transformed.obs = original_obs
        adata_transformed.var = original_var

        logger.info("Sphering transformation completed")
        return adata_transformed
    except Exception as e:
        logger.error(f"An error occurred in Sphering: {e}")
        raise