using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SocketIO;
using System;


public class SocketController : MonoBehaviour {

    public Camera frontCamera;

    private SensorManager sensorManager;
    private SocketIOComponent socket;
    private WheelDrive car;
    private Rigidbody vehicleBody;

    void Start () {
        car = GetComponent<WheelDrive>();
        sensorManager = GetComponent<SensorManager>();
        vehicleBody = GetComponent<Rigidbody>();

        socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        socket.On("car", OnCarRequest);
    }

    private void OnCarRequest(SocketIOEvent socketEvent) {
        JSONObject jsonObject = socketEvent.data;

        float angle = float.Parse(jsonObject.GetField("angle").ToString());
        float torque = float.Parse(jsonObject.GetField("torque").ToString());
        float brake = float.Parse(jsonObject.GetField("brake").ToString());


        car.SetInputAngle(angle);
        car.SetInputTorque(torque);
        car.handBrake = brake > 0.5 ? car.brakeTorque : 0;

        EmitTelemetry();
    }

    public void EmitTelemetry() {
        UnityMainThreadDispatcher.Instance().Enqueue(() => {
            Dictionary<string, JSONObject> data = new Dictionary<string, JSONObject>();

            data["angle"] = new JSONObject(car.GetInputAngle());
            data["torque"] = new JSONObject(car.GetInputTorque());
            data["brake"] = new JSONObject(car.braking);

            JSONObject aux = JSONObject.Create();
            aux.AddField("sensor0", sensorManager.GetSensorData(0));
            aux.AddField("sensor1", sensorManager.GetSensorData(1));
            aux.AddField("sensor2", sensorManager.GetSensorData(2));
            // Normalize speed (Will never reach this number, Usually 30-35 Max, Headroom)
            aux.AddField("speed", vehicleBody.velocity.magnitude / 50);
            data["aux"] = aux;

            data["image"] = JSONObject.CreateStringObject(Convert.ToBase64String(CameraHandler.RequestFrame(frontCamera)));
            socket.Emit("telemetry", new JSONObject(data));
        });
    }

}
