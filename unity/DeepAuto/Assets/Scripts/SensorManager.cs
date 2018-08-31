using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SensorManager : MonoBehaviour {

    [SerializeField]
    private float distance = 5;
    [SerializeField]
    private GameObject[] sensors;
    [SerializeField]
    private Vector3[] sensorVector;

    private float[] sensorData;

	// Update is called once per frame
	void Update () {
		for (var i = 0; i < sensors.Length; i++) {
            Transform sensorTransform = sensors[i].transform;
            Vector3 sensorDirection = sensorTransform.TransformDirection(sensorVector[i]);

            RaycastHit hit;
            if (Physics.Raycast(sensorTransform.position, sensorDirection, out hit, distance)) {
                Debug.DrawRay(sensorTransform.position, sensorDirection * hit.distance, Color.green);
            } else {
                Debug.DrawRay(sensorTransform.position, sensorDirection * distance, Color.blue);
            }
        }
	}
}
