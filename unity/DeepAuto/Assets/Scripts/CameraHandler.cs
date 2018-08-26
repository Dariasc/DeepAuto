using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class CameraHandler {

    public static int width = 800;
    public static int height = 600;

    public static byte[] RequestFrame(Camera camera) {
        // Source from Car-Sim from Udacity and NVIDIA
        RenderTexture targetTexture = camera.targetTexture;
        RenderTexture.active = targetTexture;
        Texture2D texture2D = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB24, false);
        texture2D.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
        texture2D.Apply();
        byte[] image = texture2D.EncodeToJPG();
        Object.DestroyImmediate(texture2D); // Required to prevent leaking the texture
        return image;
    }
}
