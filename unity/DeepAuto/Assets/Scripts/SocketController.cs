using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SocketIO;
using System;


public class SocketController : MonoBehaviour {

    SocketIOComponent socket;
    WheelDrive car;
    public Camera frontCamera;

	// Use this for initialization
	void Start () {
        car = GetComponent<WheelDrive>();

        socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        socket.On("car", OnCarRequest);
    }

    private void OnCarRequest(SocketIOEvent socketEvent) {
        JSONObject jsonObject = socketEvent.data;

        float angle = float.Parse(jsonObject.GetField("angle").ToString());
        float torque = float.Parse(jsonObject.GetField("torque").ToString());
        
        car.SetInputAngle(angle);
        car.SetInputTorque(torque);

        EmitTelemetry();
    }

    public void EmitTelemetry() {
        UnityMainThreadDispatcher.Instance().Enqueue(() => {
            Dictionary<string, string> data = new Dictionary<string, string>();

            data["angle"] = car.GetInputAngle().ToString();
            data["torque"] = car.GetInputTorque().ToString();

            data["image"] = Convert.ToBase64String(CameraHandler.RequestFrame(frontCamera));
            socket.Emit("telemetry", new JSONObject(data));
        });
    }

}
