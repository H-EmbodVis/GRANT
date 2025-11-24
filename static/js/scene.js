import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

let num_objects_curr = 0;
let num_objects = 100;


const layers = {
	'Toggle Name': function () {
		console.log('toggle')
		camera.layers.toggle(0);
	}
}

function onDoubleClick(event) {
	mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
	mouse.y = - (event.clientY / window.innerHeight) * 2 + 1;
	raycaster.setFromCamera(mouse, camera);
	let intersections = raycaster.intersectObjects([threejs_objects['scene0451_01']]);
	intersection = (intersections.length) > 0 ? intersections[0] : null;
	console.log(intersections);
}

function get_lines(properties) {
	var geometry = new THREE.BufferGeometry();
	let binary_filename = properties['binary_filename'];
	var positions = [];
	let num_lines = properties['num_lines'];

	fetch(binary_filename)
		.then(response => response.arrayBuffer())
		.then(buffer => {
			positions = new Float32Array(buffer, 0, 3 * num_lines * 2);
			let colors_uint8 = new Uint8Array(buffer, (3 * num_lines * 2) * 4, 3 * num_lines * 2);
			let colors_float32 = Float32Array.from(colors_uint8);
			for (let i = 0; i < colors_float32.length; i++) {
				colors_float32[i] /= 255.0;
			}
			geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
			geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors_float32, 3));
		}).then(step_progress_bar).then(render);
	var material = new THREE.LineBasicMaterial({ color: 0xFFFFFF, vertexColors: true });
	return new THREE.LineSegments(geometry, material);
}

function get_cube() {
	let cube_geometry = new THREE.BoxGeometry(1, 5, 1);
	let cube_material = new THREE.MeshPhongMaterial({ color: 0x00ffff });
	cube_material.wireframe = false;
	cube_material.wireframeLinewidth = 5;
	let cube = new THREE.Mesh(cube_geometry, cube_material);
	return cube
}

function add_progress_bar() {
	// 获取目标容器
	const container = document.getElementById("render_container");
	let gProgressElement = document.createElement("div");
	const html_code = '<div class="progress">\n' +
		'<div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%" id="progress_bar"></div>\n' +
		'</div>';
	gProgressElement.innerHTML = html_code;
	gProgressElement.id = "progress_bar_id"
	gProgressElement.style.left = "20%";
	gProgressElement.style.right = "20%";
	gProgressElement.style.position = "absolute";
	gProgressElement.style.top = "50%";


	// 把进度条添加到容器内，而不是 body
	container.appendChild(gProgressElement);
}

function step_progress_bar() {
	num_objects_curr += 1.0
	// console.log("[num_objects_curr]",num_objects_curr);
	let progress_int = parseInt(num_objects_curr / num_objects * 100.0)
	let width_string = String(progress_int) + '%';
	document.getElementById('progress_bar').style.width = width_string;
	document.getElementById('progress_bar').innerText = width_string;

	if (progress_int == 100) {
		const bar = document.getElementById('progress_bar_id');
		if (bar) {
			bar.remove();  // ✅ 彻底从页面中删除
			// console.log("[clear bar]", num_objects_curr);
		}
		// console.log("[clear bar]",num_objects_curr);
	}
}

function add_watermark() {
	let watermark = document.createElement("div");
	const html_code = '<a href="https://francisengelmann.github.io/pyviz3d/" target="_blank"><b>PyViz3D</b></a>';
	watermark.innerHTML = html_code;
	watermark.id = "watermark"
	watermark.style.right = "5px";
	watermark.style.position = "fixed";
	watermark.style.bottom = "5px";
	watermark.style.color = "#999";
	watermark.style.fontSize = "7ox";
	document.body.appendChild(watermark);
}

// -----------------------------------
// 辅助函数：计算数组的百分位数
function percentile(arr, p) {
	const sorted = arr.slice().sort((a, b) => a - b);
	const idx = Math.floor((p / 100) * sorted.length);
	return sorted[idx];
}


function get_points(properties, sceneId) {
	// Add points
	// https://github.com/mrdoob/three.js/blob/master/examples/webgl_buffergeometry_points.html
	let positions = [];
	let normals = [];
	let num_points = properties['num_points'];
	let geometry = new THREE.BufferGeometry();
	let binary_filename = `./VisAssets/${sceneId}/${properties['binary_filename']}`;

	// fetch(binary_filename)
	// 	.then(response => response.arrayBuffer())
	// 	.then(buffer => {
	// 		positions = new Float32Array(buffer, 0, 3 * num_points);
	// 		normals = new Float32Array(buffer, (3 * num_points) * 4, 3 * num_points);
	// 		let colors_uint8 = new Uint8Array(buffer, (3 * num_points) * 8, 3 * num_points);
	// 		let colors_float32 = Float32Array.from(colors_uint8);
	// 		for (let i = 0; i < colors_float32.length; i++) {
	// 			colors_float32[i] /= 255.0;
	// 		}
	// 		geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
	// 		geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
	// 		geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors_float32, 3));
	// 	})
	// 	.then(render);

	fetch(binary_filename)
		.then(response => response.arrayBuffer())
		.then(buffer => {
			positions = new Float32Array(buffer, 0, 3 * num_points);
			normals = new Float32Array(buffer, (3 * num_points) * 4, 3 * num_points);
			let colors_uint8 = new Uint8Array(buffer, (3 * num_points) * 8, 3 * num_points);
			let colors_float32 = Float32Array.from(colors_uint8);
			for (let i = 0; i < colors_float32.length; i++) {
				colors_float32[i] /= 255.0;
			}

			// -----------------------
			// 去掉高度最高的 5% 的点
			// -----------------------
			const y_values = [];
			for (let i = 0; i < num_points; i++) {
				y_values.push(positions[i * 3 + 2]); // y 轴是高度
			}
			const threshold = percentile(y_values, 80); // 95% 分位数
			const filtered_positions = [];
			const filtered_normals = [];
			const filtered_colors = [];
			for (let i = 0; i < num_points; i++) {
				if (positions[i * 3 + 2] <= threshold) {
					// 位置
					filtered_positions.push(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);
					// 法线
					filtered_normals.push(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
					// 颜色
					filtered_colors.push(colors_float32[i * 3 + 0], colors_float32[i * 3 + 1], colors_float32[i * 3 + 2]);
				}
			}

			// 使用过滤后的点创建 BufferGeometry
			geometry.setAttribute('position', new THREE.Float32BufferAttribute(filtered_positions, 3));
			geometry.setAttribute('normal', new THREE.Float32BufferAttribute(filtered_normals, 3));
			geometry.setAttribute('color', new THREE.Float32BufferAttribute(filtered_colors, 3));

			// -----------------------
			// ✅ 计算最低高度并添加网格地面
			// -----------------------
			const minY = Math.min(...filtered_positions.filter((_, i) => i % 3 === 2)); // z 轴最低值
			const gridSize = 100; // 网格大小
			const gridDivisions = 100; // 网格分段数

			const grid = new THREE.GridHelper(gridSize, gridDivisions, 0x444444, 0x888888);
			grid.rotation.x = Math.PI / 2; // 若你的高度是 z 轴，则网格应旋转以对齐
			grid.position.set(0,  0, -1.35); // 把网格放在最低点
			scene.add(grid); // ✅ 这里假设全局变量 scene 已定义
		})
		.then(render);

	let uniforms = {
		pointSize: { value: properties['point_size'] },
		alpha: { value: properties['alpha'] },
		shading_type: { value: properties['shading_type'] },
	};

	let material = new THREE.ShaderMaterial({
		uniforms: uniforms,
		vertexShader: document.getElementById('vertexshader').textContent,
		fragmentShader: document.getElementById('fragmentshader').textContent,
		transparent: true
	});

	let points = new THREE.Points(geometry, material);
	return points
}

function get_labels(properties) {
	const labels = new THREE.Group();
	labels.name = "labels"
	for (let i = 0; i < properties['labels'].length; i++) {
		const labelDiv = document.createElement('div');
		labelDiv.className = 'label';
		labelDiv.style.color = "rgb(" + properties['colors'][i][0] + ", " + properties['colors'][i][1] + ", " + properties['colors'][i][2] + ")";
		labelDiv.textContent = properties['labels'][i];

		const label_2d = new CSS2DObject(labelDiv);
		label_2d.position.set(properties['positions'][i][0], properties['positions'][i][1], properties['positions'][i][2]);
		labels.add(label_2d);
	}
	return labels
}

function get_obj(properties) {
	var container = new THREE.Object3D();
	function loadModel(object) {
		object.traverse(
			function (child) {
				if (child.isMesh) {
					let r = properties['color'][0]
					let g = properties['color'][1]
					let b = properties['color'][2]
					let colorString = "rgb(" + r + "," + g + ", " + b + ")"
					child.material.color.set(new THREE.Color(colorString));
				}
			});
		object.translateX(properties['translation'][0])
		object.translateY(properties['translation'][1])
		object.translateZ(properties['translation'][2])

		const q = new THREE.Quaternion(
			properties['rotation'][1],
			properties['rotation'][2],
			properties['rotation'][3],
			properties['rotation'][0])
		object.setRotationFromQuaternion(q)

		object.scale.set(properties['scale'][0], properties['scale'][1], properties['scale'][2])

		container.add(object)
		step_progress_bar();
		render();
	}

	const loader = new OBJLoader();
	loader.load(properties['filename'], loadModel,
		function (xhr) { // called when loading is in progresses
			console.log((xhr.loaded / xhr.total * 100) + '% loaded');
		},
		function (error) { // called when loading has errors
			console.log('An error happened');
		});
	return container
}

function get_material(alpha) {
	let uniforms = {
		alpha: { value: alpha },
		shading_type: { value: 1 },
	};
	let material = new THREE.ShaderMaterial({
		uniforms: uniforms,
		vertexShader: document.getElementById('vertexshader').textContent,
		fragmentShader: document.getElementById('fragmentshader').textContent,
		transparent: true,
	});
	return material;
}

function set_geometry_vertex_color(geometry, color) {
	const r = Math.fround(color[0] / 255.0);
	const g = Math.fround(color[1] / 255.0);
	const b = Math.fround(color[2] / 255.0);
	const num_vertices = geometry.getAttribute('position').count;
	const colors = new Float32Array(num_vertices * 3);
	for (let i = 0; i < num_vertices; i++) {
		colors[3 * i + 0] = r;
		colors[3 * i + 1] = g;
		colors[3 * i + 2] = b;
	}
	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

function get_cylinder_geometry(radius_top, radius_bottom, height, radial_segments, color) {
	let geometry = new THREE.CylinderGeometry(radius_top, radius_bottom, height, radial_segments);
	set_geometry_vertex_color(geometry, color)
	return geometry;
}

function get_sphere_geometry(radius, widthSegments, heightSegments, color) {
	const geometry = new THREE.SphereGeometry(radius, widthSegments, heightSegments);
	set_geometry_vertex_color(geometry, color);
	return geometry;
}

function get_cuboid(properties) {
	const radius_top = properties['edge_width'];
	const radius_bottom = properties['edge_width'];
	const radial_segments = 30;
	const height = 1;

	let geometry = get_cylinder_geometry(
		radius_top, radius_bottom, height, radial_segments,
		properties['color']);
	let material = get_material(properties['alpha']);

	const cylinder_x = new THREE.Mesh(geometry, material);
	cylinder_x.scale.set(1.0, properties['size'][0], 1.0)
	cylinder_x.rotateZ(3.1415 / 2.0)
	const cylinder_00 = cylinder_x.clone()
	cylinder_00.position.set(0, -properties['size'][1] / 2.0, -properties['size'][2] / 2.0)
	const cylinder_01 = cylinder_x.clone()
	cylinder_01.position.set(0, properties['size'][1] / 2.0, -properties['size'][2] / 2.0)
	const cylinder_20 = cylinder_x.clone()
	cylinder_20.position.set(0, -properties['size'][1] / 2.0, properties['size'][2] / 2.0)
	const cylinder_21 = cylinder_x.clone()
	cylinder_21.position.set(0, properties['size'][1] / 2.0, properties['size'][2] / 2.0)

	const cylinder_y = new THREE.Mesh(geometry, material);
	cylinder_y.scale.set(1.0, properties['size'][1], 1.0)
	const cylinder_02 = cylinder_y.clone()
	cylinder_02.position.set(-properties['size'][0] / 2.0, 0, -properties['size'][2] / 2.0)
	const cylinder_03 = cylinder_y.clone()
	cylinder_03.position.set(properties['size'][0] / 2.0, 0, -properties['size'][2] / 2.0)
	const cylinder_22 = cylinder_y.clone()
	cylinder_22.position.set(-properties['size'][0] / 2.0, 0, properties['size'][2] / 2.0)
	const cylinder_23 = cylinder_y.clone()
	cylinder_23.position.set(properties['size'][0] / 2.0, 0, properties['size'][2] / 2.0)

	const cylinder_z = new THREE.Mesh(geometry, material);
	cylinder_z.scale.set(1.0, properties['size'][2], 1.0)
	cylinder_z.rotateX(3.1415 / 2.0)
	const cylinder_10 = cylinder_z.clone()
	cylinder_10.position.set(-properties['size'][0] / 2.0, -properties['size'][1] / 2.0, 0.0)
	const cylinder_11 = cylinder_z.clone()
	cylinder_11.position.set(properties['size'][0] / 2.0, -properties['size'][1] / 2.0, 0.0)
	const cylinder_12 = cylinder_z.clone()
	cylinder_12.position.set(-properties['size'][0] / 2.0, properties['size'][1] / 2.0, 0.0)
	const cylinder_13 = cylinder_z.clone()
	cylinder_13.position.set(properties['size'][0] / 2.0, properties['size'][1] / 2.0, 0.0)

	let corner_geometry = get_sphere_geometry(properties['edge_width'], 30, 30, properties['color']);

	const sphere = new THREE.Mesh(corner_geometry, material);
	const corner_00 = sphere.clone()
	corner_00.position.set(-properties['size'][0] / 2.0, -properties['size'][1] / 2.0, -properties['size'][2] / 2.0)
	const corner_01 = sphere.clone()
	corner_01.position.set(properties['size'][0] / 2.0, -properties['size'][1] / 2.0, -properties['size'][2] / 2.0)
	const corner_02 = sphere.clone()
	corner_02.position.set(-properties['size'][0] / 2.0, properties['size'][1] / 2.0, -properties['size'][2] / 2.0)
	const corner_03 = sphere.clone()
	corner_03.position.set(properties['size'][0] / 2.0, properties['size'][1] / 2.0, -properties['size'][2] / 2.0)
	const corner_10 = sphere.clone()
	corner_10.position.set(-properties['size'][0] / 2.0, -properties['size'][1] / 2.0, properties['size'][2] / 2.0)
	const corner_11 = sphere.clone()
	corner_11.position.set(properties['size'][0] / 2.0, -properties['size'][1] / 2.0, properties['size'][2] / 2.0)
	const corner_12 = sphere.clone()
	corner_12.position.set(-properties['size'][0] / 2.0, properties['size'][1] / 2.0, properties['size'][2] / 2.0)
	const corner_13 = sphere.clone()
	corner_13.position.set(properties['size'][0] / 2.0, properties['size'][1] / 2.0, properties['size'][2] / 2.0)

	const cuboid = new THREE.Group();
	cuboid.position.set(properties['position'][0], properties['position'][1], properties['position'][2])
	cuboid.add(cylinder_00)
	cuboid.add(cylinder_01)
	cuboid.add(cylinder_20)
	cuboid.add(cylinder_21)
	cuboid.add(cylinder_02)
	cuboid.add(cylinder_03)
	cuboid.add(cylinder_22)
	cuboid.add(cylinder_23)
	cuboid.add(cylinder_10)
	cuboid.add(cylinder_11)
	cuboid.add(cylinder_12)
	cuboid.add(cylinder_13)

	cuboid.add(corner_00)
	cuboid.add(corner_01)
	cuboid.add(corner_02)
	cuboid.add(corner_03)
	cuboid.add(corner_10)
	cuboid.add(corner_11)
	cuboid.add(corner_12)
	cuboid.add(corner_13)

	const q = new THREE.Quaternion(
		properties['orientation'][1],
		properties['orientation'][2],
		properties['orientation'][3],
		properties['orientation'][0])
	cuboid.setRotationFromQuaternion(q)
	cuboid.position.set(properties['position'][0], properties['position'][1], properties['position'][2])
	return cuboid
}

function get_polyline(properties) {
	const radius_top = properties['edge_width']
	const radius_bottom = properties['edge_width']
	const radial_segments = 5;
	const height = 1;
	let material = get_material(properties['alpha']);
	let geometry = get_cylinder_geometry(radius_top, radius_bottom, height, radial_segments, properties['color']);
	const cylinder = new THREE.Mesh(geometry, material);
	let corner_geometry = get_sphere_geometry(properties['edge_width'], radial_segments, radial_segments, properties['color']);
	const sphere = new THREE.Mesh(corner_geometry, material);
	const polyline = new THREE.Group();

	// Add first corner to the polyline
	const corner_0 = sphere.clone()
	corner_0.position.set(properties['positions'][0][0], properties['positions'][0][1], properties['positions'][0][2])
	polyline.add(corner_0)

	for (var i = 1; i < properties['positions'].length; i++) {
		// Put the sphere the make a nice round corner
		const corner_i = sphere.clone()
		corner_i.position.set(properties['positions'][i][0],
			properties['positions'][i][1],
			properties['positions'][i][2])

		// Put a segment connecting the two last points
		const cylinder_i = cylinder.clone()
		var dist_x = properties['positions'][i][0] - properties['positions'][i - 1][0]
		var dist_y = properties['positions'][i][1] - properties['positions'][i - 1][1]
		var dist_z = properties['positions'][i][2] - properties['positions'][i - 1][2]
		var cylinder_length = Math.sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z)
		cylinder_i.scale.set(1.0, cylinder_length, 1.0)
		cylinder_i.lookAt(properties['positions'][i][0] - properties['positions'][i - 1][0],
			properties['positions'][i][1] - properties['positions'][i - 1][1],
			properties['positions'][i][2] - properties['positions'][i - 1][2])
		cylinder_i.rotateX(3.1415 / 2.0)
		cylinder_i.position.set(properties['positions'][i - 1][0],
			properties['positions'][i - 1][1],
			properties['positions'][i - 1][2])
		cylinder_i.translateY(cylinder_length / 2.0)
		polyline.add(cylinder_i)
	}

	return polyline
}

function get_arrow(properties) {
	const radius_top = 0.0;
	const radius_bottom = properties['head_width'];
	const radial_segments = 15;
	const height = radius_bottom * 2.0;

	var dist_x = properties['end'][0] - properties['start'][0]
	var dist_y = properties['end'][1] - properties['start'][1]
	var dist_z = properties['end'][2] - properties['start'][2]
	var margnitude = Math.sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z)

	let material = get_material(properties['alpha']);
	let geometry = get_cylinder_geometry(radius_top, radius_bottom, height, radial_segments, properties['color']);
	let geometry_stroke = get_cylinder_geometry(properties['stroke_width'], properties['stroke_width'], margnitude - height, radial_segments, properties['color']);

	const arrow_head = new THREE.Mesh(geometry, material);
	arrow_head.translateY(margnitude - height / 2.0)
	const arrow_stroke = new THREE.Mesh(geometry_stroke, material);
	arrow_stroke.translateY(margnitude / 2.0 - height / 2.0)

	const arrow = new THREE.Group();
	arrow.add(arrow_head);
	arrow.add(arrow_stroke);

	arrow.lookAt(properties['end'][0] - properties['start'][0],
		properties['end'][1] - properties['start'][1],
		properties['end'][2] - properties['start'][2])
	arrow.rotateX(3.1415 / 2.0)
	arrow.position.set(properties['start'][0], properties['start'][1], properties['start'][2])
	return arrow;
}

// function get_ground() {
// 	let mesh = new THREE.Mesh(new THREE.PlaneBufferGeometry(2000, 2000),
// 		new THREE.MeshPhongMaterial({ color: 0x999999, depthWrite: true }));
// 	mesh.rotation.x = -Math.PI / 2;
// 	mesh.position.set(0, -5, 0);
// 	mesh.receiveShadow = true;
// 	return mesh;
// }
function get_ground() {
	const size = 2000;
	const divisions = 40;
	const gridHelper = new THREE.GridHelper(size, divisions, 0x888888, 0xcccccc);
	gridHelper.position.y = -5;
	return gridHelper;
}



function init_gui(objects) {
	let menuMap = new Map();
	for (const [name, value] of Object.entries(objects)) {
		let splits = name.split(';');
		if (splits.length > 1) {
			let folder_name = splits[0];
			if (!menuMap.has(folder_name)) {
				menuMap.set(folder_name, gui.addFolder(folder_name));
			}
			let fol = menuMap.get(folder_name);
			fol.add(value, 'visible').name(splits[1]).onChange(render);
			fol.open();
		} else {
			if (value.name.localeCompare('labels') != 0) {
				gui.add(value, 'visible').name(name).onChange(render);
			}
		}
	}
}

function render() {
	renderer.render(scene, camera);
	labelRenderer.render(scene, camera);
}

function init() {
	scene.background = new THREE.Color(0xffffff); //  0xffffff
	const width = container.clientWidth;
	const height = container.clientHeight;
	renderer.setSize(width, height);
	labelRenderer.setSize(width, height);

	let hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
	hemiLight.position.set(0, 20, 0);
	// scene.add(hemiLight);
	// scene.add(get_ground());

	let dirLight = new THREE.DirectionalLight(0xffffff);
	dirLight.position.set(-10, 10, - 10);
	dirLight.castShadow = true;
	dirLight.shadow.camera.top = 2;
	dirLight.shadow.camera.bottom = - 2;
	dirLight.shadow.camera.left = - 2;
	dirLight.shadow.camera.right = 2;
	dirLight.shadow.camera.near = 0.1;
	dirLight.shadow.camera.far = 40;
	//scene.add(dirLight);

	let intensity = 0.5;
	let color = 0xffffff;
	const spotLight1 = new THREE.SpotLight(color, intensity);
	spotLight1.position.set(100, 1000, 0);
	scene.add(spotLight1);
	const spotLight2 = new THREE.SpotLight(color, intensity / 3.0);
	spotLight2.position.set(100, -1000, 0);
	scene.add(spotLight2);
	const spotLight3 = new THREE.SpotLight(color, intensity);
	spotLight3.position.set(0, 100, 1000);
	scene.add(spotLight3);
	const spotLight4 = new THREE.SpotLight(color, intensity / 3.0);
	spotLight4.position.set(0, 100, -1000);
	scene.add(spotLight4);
	const spotLight5 = new THREE.SpotLight(color, intensity);
	spotLight5.position.set(1000, 0, 100);
	scene.add(spotLight5);
	const spotLight6 = new THREE.SpotLight(color, intensity / 3.0);
	spotLight6.position.set(-1000, 0, 100);
	scene.add(spotLight6);

	raycaster = new THREE.Raycaster();
	raycaster.params.Points.threshold = 1.0;



}


function add_threejs_objects_to_scene(threejs_objects) {
	for (const [key, value] of Object.entries(threejs_objects)) {
		scene.add(value);
	}


}

function onWindowResize() {
	const container = document.getElementById('render_container');
	const width = container.clientWidth;
	const height = container.clientHeight;

	renderer.setSize(width, height);
	labelRenderer.setSize(width, height);

	camera.aspect = width / height;
	camera.updateProjectionMatrix();
	render();
}



function loadScene(sceneId) {
	// 清空旧场景（如已有 mesh、GUI、progress bar）
	clear_threejs_scene();        // 自定义函数，清除场景中对象
	remove_progress_bar();        // 若你有该函数，清除旧进度条
	// clear_gui();

	fetch(`./VisAssets/${sceneId}/nodes.json`)
		.then(response => { add_progress_bar(); return response; })
		.then(json_response => { console.log(json_response); return json_response }) // 可以成功读取
		.then(response => response.json())
		.then(json => {
			console.log(`[Scene: ${sceneId}] loaded`, json);
			return create_threejs_objects(json, sceneId); // 这里会更新controls
		})
		.then(() => add_threejs_objects_to_scene(threejs_objects))
		// .then(() => init_gui(threejs_objects))

		.then(() => init_task_gui(threejs_objects, sceneId))
		.then(() => console.log("Done loading " + sceneId))
		.then(render)
		.catch(err => console.error("Failed to load scene:", err));
}

function /* The above code is a comment in JavaScript indicating that the following code will involve
creating Three.js objects. */
	/* The above code is a comment in JavaScript indicating that the following code will create
	Three.js objects. */
	create_threejs_objects(properties, sceneId) {
	threejs_objects = {};
	num_objects_curr = 0.0;
	num_objects = parseFloat(Object.entries(properties).length);
	console.log("[num_objects]", num_objects);
	for (const [object_name, object_properties] of Object.entries(properties)) {
		if (String(object_properties['type']).localeCompare('camera') == 0) {
			console.log("[Camera Object Detected]", object_name, object_properties); // ✅ 打印对象信息
			set_camera_properties(object_properties);
			render();
			console.log("[step from camera]"); // ✅ 打印对象信息
			step_progress_bar();
			continue;
		}
		if (String(object_properties['type']).localeCompare('points') == 0) {
			threejs_objects[object_name] = get_points(object_properties, sceneId); // 加载点云的数据
			step_progress_bar(); // 加载数据
			console.log("[step from points]"); // 这里调用的次数是对的
			render();
		}
		if (String(object_properties['type']).localeCompare('labels') == 0) {
			threejs_objects[object_name] = get_labels(object_properties);
			step_progress_bar(); // 加载数据
			render();
		}
		if (String(object_properties['type']).localeCompare('lines') == 0) {
			threejs_objects[object_name] = get_lines(object_properties);
			render();
		}
		if (String(object_properties['type']).localeCompare('obj') == 0) {
			threejs_objects[object_name] = get_obj(object_properties);
		}
		if (String(object_properties['type']).localeCompare('cuboid') == 0) {
			threejs_objects[object_name] = get_cuboid(object_properties);
			step_progress_bar();
			render();
		}
		if (String(object_properties['type']).localeCompare('polyline') == 0) {
			threejs_objects[object_name] = get_polyline(object_properties);
			step_progress_bar();
			render();
		}
		if (String(object_properties['type']).localeCompare('arrow') == 0) {
			threejs_objects[object_name] = get_arrow(object_properties);
			step_progress_bar();
			render();
		}
		threejs_objects[object_name].visible = object_properties['visible'];
		threejs_objects[object_name].frustumCulled = false;
	}

	// Add axis helper
	threejs_objects['Axis'] = new THREE.AxesHelper(1);

	render();
}



function set_camera_properties(properties) {
	camera.up.set(properties['up'][0],
		properties['up'][1],
		properties['up'][2]);
	camera.position.set(properties['position'][0],
		properties['position'][1],
		properties['position'][2]);
	update_controls();

	controls.update();
	controls.target = new THREE.Vector3(properties['look_at'][0],
		properties['look_at'][1],
		properties['look_at'][2]);
	camera.updateProjectionMatrix();
	controls.update();
}

function update_controls() {
	if (controls) {
		controls.dispose(); // 移除事件绑定
	}
	controls = new OrbitControls(camera, labelRenderer.domElement);
	controls.zoomSpeed = 0.1; // 缩小滚轮灵敏度
	controls.addEventListener("change", render);
	controls.enableKeys = true;
	controls.enablePan = false; // enable dragging

	const domElement = labelRenderer.domElement;

	// domElement.addEventListener('mousemove', (event) => {
	// 		if (event.ctrlKey) {
	// 			controls.enablePan = true;
	// 			controls.enableRotate = false;
	// 		} else {
	// 			controls.enablePan = false;
	// 			controls.enableRotate = true;
	// 		}


	// 	});

	// domElement.addEventListener('mousedown', (event) => {
	// 	if (event.ctrlKey) {
	// 		controls.enableRotate = false;
	// 		controls.enablePan = true;
	// 	} else {
	// 		controls.enableRotate = true;
	// 		controls.enablePan = false;
	// 	}
	// });

	// domElement.addEventListener('mouseup', () => {
	// 	// 恢复默认（你可以自定义恢复行为）
	// 	controls.enableRotate = true;
	// 	controls.enablePan = true;
	// });

domElement.addEventListener('mousedown', (event) => {
	if (event.button === 2) { // 右键
		controls.enableRotate = false;
		controls.enablePan = true;
	} else {
		controls.enableRotate = true;
		controls.enablePan = false;
	}
});

domElement.addEventListener('mouseup', () => {
	// 恢复默认（你可以自定义恢复行为）
	controls.enableRotate = true;
	controls.enablePan = true;
});

// 防止右键弹出浏览器菜单
domElement.addEventListener('contextmenu', (event) => {
	event.preventDefault();
});

}


function clear_threejs_scene() {
	while (scene.children.length > 0) {
		scene.remove(scene.children[0]);
	}
}

function clear_gui() {
	if (gui) {
		gui.destroy();  // lil-gui 的销毁方法
		gui = new GUI({ autoPlace: false, width: 200 });
		document.getElementById("render_container").appendChild(gui.domElement);
	}
}

function remove_progress_bar() {
	const pbar = document.getElementById("progress_bar_id");
	if (pbar) pbar.remove();
}
// let queryData = [];
// fetch(`./VisAssets/scene1/query.json`)
// .then(response => response.json())
// .then(json => queryData = json);

async function init_task_gui(objects, sceneId) {
	// 加载 query.json（一次即可）
	const response = await fetch(`./VisAssets/${sceneId}/query.json`);
	const queryData = await response.json();

	const container = document.getElementById('task_gui_container');
	container.innerHTML = ''; // 清空旧内容

	// Step 1: 按照 task_id 分组
	const taskMap = new Map();

	for (const [name, value] of Object.entries(objects)) {
		let [task_id, action_name] = name.split(';');

		// ✅ 跳过不包含 "task" 的项
		if (!task_id || !task_id.includes("task") || !action_name.includes("dataset")) continue;

		if (!taskMap.has(task_id)) {
			taskMap.set(task_id, []);
		}
		taskMap.get(task_id).push({ action_name, object: value });
	}

	// Step 2: 创建 task_id 下拉表单
	// Step 2: 创建 task_id 下拉表单

	const select = document.createElement('select');
// 隐藏原生箭头（兼容大部分浏览器）


	select.style.padding = '10px 16px';
	select.style.borderRadius = '10px';
	select.style.border = 'none';
	select.style.background = 'linear-gradient(135deg, #a8c8ff, #6fa8ff)';
	select.style.color = '#fff';
	select.style.fontWeight = '500';
	select.style.cursor = 'pointer';
	select.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
	select.style.marginBottom = '12px';
	select.style.transition = 'all 0.2s ease';
	select.style.outline = 'none';
	select.style.fontSize = '16px';   // 增大字体
	select.style.fontWeight = '600';  // 增加粗度
	select.style.width = '100%';  // 宽度沾满父容器
	select.style.textAlign = 'center'; // 字体居中


	// 悬停效果
	select.addEventListener('mouseenter', () => {
		select.style.background = 'linear-gradient(135deg, #c0d9ff, #8bb6ff)';
	});
	select.addEventListener('mouseleave', () => {
		select.style.background = 'linear-gradient(135deg, #a8c8ff, #6fa8ff)';
	});

	// 添加 option
	for (const task_id of taskMap.keys()) {
		const opt = document.createElement('option');
		opt.value = task_id;
		const taskMatch = task_id.match(/\d+/);
		const taskIdstr = parseInt(taskMatch[0]);
		if (taskIdstr >= 3) continue; // ✅ 只保留 taskIdstr < 3 的任务
		opt.textContent = `Task ${taskIdstr + 1}`;
		opt.style.color = '#333'; // option 内文字颜色
		select.appendChild(opt);
	}

	container.appendChild(select);


	// Step 3: 创建 action checkbox 容器
	const actionsDiv = document.createElement('div');
	// actionsDiv.id = 'actions_div';
	// actionsDiv.style.backgroundColor = 'lightblue';  
	container.appendChild(actionsDiv);

	// Step 4: 当 task_id 改变时，更新对应的 action 列表
	select.addEventListener('change', () => {
		updateActions(select.value);
	});

	// 初始化时加载第一个 task 的 action
	updateActions(select.value);

	function updateActions(task_id) {

		// ✅ 先将所有 object 设置为不可见
		for (const actions of taskMap.values()) {
			for (const { object } of actions) {
				object.visible = false;
			}
		}

		actionsDiv.innerHTML = ''; // 清空旧的 action 控件
		// ✅ 3. 加载 query.json，找到与 task_id 匹配、action_id 为 100 的条目

		// // 在 queryData 中查找匹配项
		// for (const item of queryData) {
		// 	if (item.name === `task_id: ${taskId}, action_id: ${actionId}`) {
		// 		// console.log("find is ", item.text);
		// 		return item.text;
		// 	}
		// 	// console.log("dont find!");
		// }
		const taskMatch = task_id.match(/\d+/);             // 提取 task 中的数字
		const taskId = parseInt(taskMatch[0]);
		const queryItem = queryData.find(item => item.name === `Adataset_task_id: ${taskId}, task_info`);
		console.log("task queryData is", queryData);
		console.log("task queryItem is", queryItem);

		if (queryItem) {
			// 	// ✅ 4. 显示 question 框
			// 	const questionBox = document.createElement('div');
			// 	questionBox.textContent = queryItem.question;
			// 	questionBox.style.display = 'inline-block';
			// 	questionBox.style.backgroundColor = 'rgba(145, 180, 255, 0.2)';
			// 	questionBox.style.padding = '8px 12px';
			// 	questionBox.style.marginBottom = '10px';
			// 	questionBox.style.border = '1px solid #999';
			// 	questionBox.style.borderRadius = '6px';
			// 	questionBox.style.whiteSpace = 'pre-wrap'; // 保留换行
			// 	actionsDiv.appendChild(questionBox);
			// }
			// ✅ 4. 显示 question 框（美化版）
			const questionBox = document.createElement('div');
			questionBox.textContent = queryItem.question;
			questionBox.style.display = 'inline-block';
			questionBox.style.padding = '10px 16px';
			questionBox.style.marginBottom = '12px';
			questionBox.style.borderRadius = '10px';
			questionBox.style.whiteSpace = 'pre-wrap';
			questionBox.style.fontSize = '15px';
			questionBox.style.fontWeight = '500';
			questionBox.style.color = '#f9f9ff';
			questionBox.style.background = 'linear-gradient(135deg, #a8c8ff, #6fa8ff)';
			questionBox.style.boxShadow = '0 3px 8px rgba(0, 0, 0, 0.1)';
			questionBox.style.border = '1px solid rgba(255, 255, 255, 0.2)';
			questionBox.style.backdropFilter = 'blur(6px)'; // 玻璃感
			questionBox.style.transition = 'transform 0.2s ease, box-shadow 0.2s ease';
			questionBox.style.border = '1px solid rgba(255, 255, 255, 0.25)';
			questionBox.style.fontWeight = '600';  // 中等粗细

			// 悬停时轻微浮起
			questionBox.addEventListener('mouseenter', () => {
				questionBox.style.transform = 'translateY(-2px)';
				questionBox.style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.25)';
			});
			questionBox.addEventListener('mouseleave', () => {
				questionBox.style.transform = 'translateY(0)';
				questionBox.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.15)';
			});

			actionsDiv.appendChild(questionBox);
		}



		const actions = taskMap.get(task_id);

		actions.forEach(({ action_name, object }, index) => {
			const label = document.createElement('label');

			label.style.display = 'inline-block';             // ✅ 使边框自适应文字
			label.style.backgroundColor = 'rgba(211, 255, 233, 0.2)';       // ✅ 初始浅绿色
			label.style.padding = '4px 8px';                  // ✅ 添加内边距美观
			label.style.marginBottom = '6px';                 // ✅ 控件间距
			label.style.border = '1px solid #999';
			label.style.borderRadius = '6px';
			label.style.cursor = 'pointer';
			// 添加鼠标悬停效果
			label.style.transition = 'all 0.2s ease';

			label.style.background = 'linear-gradient(135deg, #d0f5e0, #a8e8c8)'; // 浅绿色渐变
			label.style.border = 'none';
			label.style.fontWeight = '500';  // 中等粗细
			label.style.color = '#333333';  // 浅黑色文字，更柔和
			label.addEventListener('mouseenter', () => {
				if (!checkbox.checked) {
					label.style.background = 'linear-gradient(135deg, #c7f7de, #91e8bf)';
				}
				label.style.transform = 'translateY(-2px)';
				label.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
			});

			label.addEventListener('mouseleave', () => {
				if (!checkbox.checked) {
					label.style.background = 'linear-gradient(135deg, #b6f3d5, #7edfb2)';
				}
				label.style.transform = 'translateY(0)';
				label.style.boxShadow = 'none';
			});
			// label.addEventListener('mouseenter', () => {
			// 	if (!checkbox.checked) {
			// 		label.style.backgroundColor = 'rgba(211, 255, 233, 0.2)'; // 悬停色
			// 	}
			// 	label.style.transform = 'translateY(-2px)';
			// 	label.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
			// });

			// label.addEventListener('mouseleave', () => {
			// 	if (!checkbox.checked) {
			// 		label.style.backgroundColor = 'rgba(211, 255, 233, 0.2)'; // 恢复原色
			// 	}
			// 	label.style.transform = 'translateY(0)';
			// 	label.style.boxShadow = 'none';
			// });



			const checkbox = document.createElement('input');
			checkbox.type = 'checkbox';
			checkbox.name = 'action';
			checkbox.value = action_name;
			checkbox.style.marginRight = '6px';




			checkbox.addEventListener('change', () => {
				// 全部隐藏本组中的 object
				for (const { object: obj } of actions) {
					obj.visible = false;
				}

				// 当前勾选 → 显示当前 object，变深绿色；取消 → 隐藏
				if (checkbox.checked) {
					object.visible = true;

					// 取消其他 checkbox 勾选状态
					const allCheckboxes = actionsDiv.querySelectorAll('input[type="checkbox"]');
					allCheckboxes.forEach(cb => {
						if (cb !== checkbox) cb.checked = false;
					});

					// 设置所有 label 背景为浅渐变绿色，字体黑色
					const allLabels = actionsDiv.querySelectorAll('label');
					allLabels.forEach(lb => {
						lb.style.background = 'linear-gradient(135deg, #d0f5e0, #a8e8c8)';
						lb.style.color = '#333333';
					});

					// 当前 label 背景加深渐变，字体白色
					label.style.background = 'linear-gradient(135deg, #76d9a8, #39b37a)'; // 更深的绿色
					label.style.color = '#ffffff';
						// ✅ 聚焦到该 object
	focusOnObject(object);

				} else {
					object.visible = false;
					label.style.background = 'linear-gradient(135deg, #d0f5e0, #a8e8c8)';
					label.style.color = '#333333';
				}


				render();
			});

			// ✅ 使用 queryData 查找更好看的描述文本
			const readableText = findTextFromQuery(task_id, action_name, queryData);

			label.appendChild(checkbox);
			label.appendChild(document.createTextNode(` ${readableText}`));
			actionsDiv.appendChild(label);
		});
		// ✅ 5. 显示 solution 框
		// const solutionBox = document.createElement('div');
		// solutionBox.textContent = queryItem.thinking;
		// solutionBox.style.display = 'inline-block';
		// solutionBox.style.backgroundColor = 'rgba(224, 145, 255, 0.2)';
		// solutionBox.style.padding = '8px 12px';
		// solutionBox.style.marginBottom = '16px';
		// solutionBox.style.border = '1px solid #999';
		// solutionBox.style.borderRadius = '6px';
		// solutionBox.style.whiteSpace = 'pre-wrap';
		// actionsDiv.appendChild(solutionBox);
	}
}


function focusOnObject(object) {
	// 计算物体的包围盒中心
	const box = new THREE.Box3().setFromObject(object);
	const center = box.getCenter(new THREE.Vector3());
	const size = box.getSize(new THREE.Vector3());

	// 根据物体大小调整相机距离
	const maxDim = Math.max(size.x, size.y, size.z);
	const fitDistance = maxDim * 3;

	// 相机当前位置 → 目标位置
	const direction = new THREE.Vector3()
		.subVectors(camera.position, controls.target)
		.normalize()
		.multiplyScalar(fitDistance);

	const newCameraPos = new THREE.Vector3().addVectors(center, direction);

	// 平滑过渡（非瞬移）
	const tweenDuration = 600; // 毫秒
	const startPos = camera.position.clone();
	const startTarget = controls.target.clone();
	const startTime = performance.now();

	function animateFocus() {
		const elapsed = performance.now() - startTime;
		const t = Math.min(elapsed / tweenDuration, 1);
		// 线性插值
		camera.position.lerpVectors(startPos, newCameraPos, t);
		controls.target.lerpVectors(startTarget, center, t);
		controls.update();
		render();
		if (t < 1) requestAnimationFrame(animateFocus);
	}
	animateFocus();
}

// function focusOnObject(object) {
// 	// 计算物体的包围盒中心
// 	const box = new THREE.Box3().setFromObject(object);
// 	const center = box.getCenter(new THREE.Vector3());
// 	const size = box.getSize(new THREE.Vector3());

// 	// 计算相机应离中心多远
// 	const maxDim = Math.max(size.x, size.y, size.z);
// 	const fitDistance = maxDim * 3.5; // 可自行调整，如 maxDim * 3.5

// 	// -------------------------
// 	// 🚩 关键改动：
// 	// 让相机从“原点→物体中心”的方向看过去
// 	// -------------------------
// 	const direction = new THREE.Vector3()
// 		.subVectors( new THREE.Vector3(0, 0, 1.5), center) // 从原点指向物体中心
// 		.normalize();

// 	// 相机的新位置 = 物体中心 + （反方向 * 距离）
// 	const newCameraPos = new THREE.Vector3()
// 		.copy(center)
// 		.addScaledVector(direction, fitDistance);

// 	// 平滑过渡（非瞬移）
// 	const tweenDuration = 800; // 毫秒
// 	const startPos = camera.position.clone();
// 	const startTarget = controls.target.clone();
// 	const startTime = performance.now();

// 	function animateFocus() {
// 		const elapsed = performance.now() - startTime;
// 		const t = Math.min(elapsed / tweenDuration, 1);

// 		// 线性插值
// 		camera.position.lerpVectors(startPos, newCameraPos, t);
// 		controls.target.lerpVectors(startTarget, center, t);
// 		controls.update();
// 		render();

// 		if (t < 1) requestAnimationFrame(animateFocus);
// 	}
// 	animateFocus();
// }




// queryData 是 query.json 加载后的数组
function findTextFromQuery(task_id_str, action_name_str, queryData) {
	// console.log("queryData", queryData);
	const taskMatch = task_id_str.match(/\d+/);             // 提取 task 中的数字
	const actionMatch = action_name_str.match(/dataset_action_(\d+)/); // 提取 action 中的数字

	const taskId = parseInt(taskMatch[0]);
	const actionId = parseInt(actionMatch[1]);
	console.log("taskMatch is ", taskId);
	console.log("actionMatch is ", actionId);
	// if (taskMatch && actionMatch && parseInt(actionMatch[1]) == 100){
	// 	if (item.name === `task_id: ${taskId}, action_id: ${actionId}`) {
	// 		console.log("task is ", item.text);
	// 		return item.question;
	// 	}
	// } 

	if (!taskMatch || !actionMatch) {
		console.log("actionMatch is ", actionId);
		return action_name_str; // fallback：原始名称
	}




	// 在 queryData 中查找匹配项
	for (const item of queryData) {
		if (item.name === `Adataset_task_id: ${taskId}, dataset_action_id: ${actionId}`) {
			// console.log("find is ", item.text);
			return item.text;
		}
		// console.log("dont find!");
	}

	// 找不到则返回原名
	return action_name_str;
}






const scene = new THREE.Scene();
// 获取容器元素
const container = document.getElementById('render_container');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight); // ✅ 设置为容器尺寸
container.appendChild(renderer.domElement);

// 初始化 camera
const camera = new THREE.PerspectiveCamera(
	75, // 垂直反向视野角度
	container.clientWidth / container.clientHeight, // ✅ 容器比例，纵宽比例
	0.01, // 近裁剪面
	1000 // 远裁剪面
);
camera.updateProjectionMatrix();
var controls = '';

let labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0px';
document.getElementById('render_container').appendChild(labelRenderer.domElement)

window.addEventListener('resize', onWindowResize, false);

let raycaster;
let intersection = null;
let mouse = new THREE.Vector2();

let threejs_objects = {};

init();


document.addEventListener("DOMContentLoaded", () => {
	const thumbs = document.querySelectorAll(".scene-thumb");

	thumbs.forEach((thumb) => {
		thumb.addEventListener("click", () => {
			thumbs.forEach((t) => t.classList.remove("active"));
			thumb.classList.add("active");

			const sceneId = thumb.getAttribute("data-value");
			loadScene(sceneId); // 加载对应的 scene
		});
	});

	// 默认加载第一个 scene
	if (thumbs.length > 0) {
		const firstScene = thumbs[0].getAttribute("data-value");
		thumbs[0].classList.add("active");
		loadScene(firstScene);
	}
});