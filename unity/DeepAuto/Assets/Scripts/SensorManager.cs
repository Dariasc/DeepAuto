using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SensorManager : MonoBehaviour {

    [SerializeField]
    private float distance = 5;
    [SerializeField]
    private GameObject[] sensors;

    private float[] sensorData = new float[3];

	// Update is called once per frame
	void Update () {
		for (var i = 0; i < sensors.Length; i++) {
            Sensor sensor = sensors[i].GetComponent<Sensor>();

            Transform sensorTransform = sensors[i].transform;
            Vector3 sensorDirection = sensorTransform.TransformDirection(sensor.direction);

            RaycastHit hit;
            if (Physics.Raycast(sensorTransform.position, sensorDirection, out hit, distance)) {
                Debug.DrawRay(sensorTransform.position, sensorDirection * hit.distance, Color.green);
            } else {
                Debug.DrawRay(sensorTransform.position, sensorDirection * distance, Color.blue);
            }
            sensorData[i] = hit.distance;
        }
    }

    public float GetSensorData(int i) {
        return sensorData[i] / distance;
    }

}
