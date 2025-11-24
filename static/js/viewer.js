// let camera, scene, renderer, controls, breathe=true, tween, looktween;
// let breath_depth=0, breath_speed=0;
// let renderer_seg, scene_seg;
// var cam_x=2, cam_y=2, cam_z=2, delta=0;
// let target_size = 0;
// let view_time = 2500;
// let scene_graph;
// let distance = 8;
// let rotation_r = 0;
// var axesHelper = new THREE.AxesHelper(20);
// var gridHelper = new THREE.GridHelper(20, 20);
// const startPos = new THREE.Vector3(0, 0, 0); 
// const endPos = new THREE.Vector3(3, 3, 3);
// const clock = new THREE.Clock();
// let action_tween;
// let scene_datas = {};
// let selected_scenes_list = false;
// let object_mesh_names = [];
// let local_data;

// const type_mapping = {
//     "rewrite": "rewritten description",
//     "template": "template-based description",
//     "anno": "annotated description",
//     "caption": "object-centric caption",
//     "default": "",
// }
// const type_message_mapping = {
//     "scene_cap": "Scene Caption: ",
//     "rewrite": "LLM-refined: ",
//     "template": "Template-based: ",
//     "anno": "Annotation: ",
//     "obj_cap": "Object Caption: ",
//     "default": "",
// };
// const modeList = ["obj_cap", "anno", "rewrite", "template"]

// // get scene data 
// async function loadLocalData() {
//     try {
//       const response = await fetch("./assets/datas/sample_seq.json");
//       scene_datas = await response.json();
//       console.log("Data loaded:", scene_datas);
//     } catch (error) {
//       console.error("Error loading local data:", error);
//     }
// }

// function setScrollPosition() {
//     const chatBody = document.querySelector(".chat-body");
//     if (chatBody.scrollHeight > 0) {
//         chatBody.scrollTop = chatBody.scrollHeight;
//     }
// };

// function processMessage(message, type="default") {
//     if (message instanceof String) {
//         return message;
//     } else {
//         if (message == null || message.length == 0) {
//             if (type == "default") {
//                 return "No available description for this object.";
//             } else {
//                 let type_name = type_mapping[type];
//                 return `No available ${type_name} for this object.`
//             }
//         } else {
//             // temporarily returing the first one 
//             highlight_message = type_message_mapping[type] + message[0];
//             console.log(message.length, highlight_message, highlight_message.length)
//             return highlight_message;
//         }
//     }
// };

// function renderMessageEle(txt, type){
//     var type_ch = document.getElementById(type);
//     if (type == "scene_cap" || type_ch.checked) {
//         const chatBody = document.querySelector(".chat-body");
//         let className = "chatbot-message";
//         const messageEle = document.createElement("span");
//         messageEle.classList.add(className);
//         messageEle.classList.add(type);
//         chatBody.append(messageEle);
//         typeWriter(messageEle, 0, txt);
//         return txt.length * 10;
//     } else {
//         return 0;
//     }
// };

// function renderMessageEleSeq(data, type) {
//     var type_ch = document.getElementById(type);
//     if (type == "seq_grounding" || type_ch.checked) {
//         const chatBody = document.querySelector(".chat-body");
//         let className = "chatbot-message";
//         // Handle seq_grounding data
//         data.forEach((item, index) => {
//             const taskEle = document.createElement("div");
//             taskEle.classList.add(className);
//             taskEle.classList.add(type);
//             taskEle.style.fontSize = "16px";
//             taskEle.style.fontWeight = "bold";
//             taskEle.style.backgroundColor = "#87CEEB";  // sky blue
//             taskEle.style.border = "double";
//             taskEle.textContent = `Task ${index + 1}: ${item.task_description}`;
//             chatBody.append(taskEle);
//             item.action_steps.forEach(step => {
//                 const stepEle = document.createElement("button");
//                 stepEle.classList.add(className);
//                 stepEle.classList.add(type);
//                 stepEle.style.textAlign = "left";
//                 stepEle.style.fontSize = "16px";
//                 stepEle.style.fontWeight = "normal";
//                 stepEle.style.backgroundColor = "#FFE066";  // light gold
//                 stepEle.style.color = "black";
//                 stepEle.style.border = "outset";
//                 stepEle.textContent = step.action + " (" + step.label + ")";
//                 stepEle.addEventListener("click", function() {
//                     console.log("Step element clicked!");
//                     camera_look_target(step.size, step.position, 2000);
//                     minBbox = box_debug(step.position, step.size, 10);
//                     if (minBbox != null) {
//                         scene.add(minBbox);
//                         tweencamera.start();
//                         // cleanChat();
//                         // console.log("calling caption:")
//                         console.log(minBbox)
//                         console.log(step)
//                         // get_caption_mode(step.caption, 0, modeList);
//                     }
//                 });
//                 chatBody.append(stepEle);
//             });
//         });
//     }
// }

// function cleanChat() {
//     let e = document.querySelector(".chat-body");
//     e.innerHTML = "";
// }

// function typeWriter(container, charIndex, dialogueText) {
//     if (charIndex < dialogueText.length) {
//         container.innerHTML += dialogueText.charAt(charIndex);
//         charIndex++;
//         setTimeout(typeWriter, 4, container, charIndex, dialogueText); // Adjust typing speed (milliseconds)
//     }
//     setScrollPosition();
// }

// function get_caption_mode(object_caption, modeIndex, modeList){
//     // cleanChat();
//     console.log(modeList[modeIndex], modeIndex);
//     if (modeIndex < modeList.length) {
//         type = modeList[modeIndex];
//         console.log("calling renderMessageEle");
//         delay_time = renderMessageEle(processMessage(object_caption[type], type), type);
//         modeIndex++;
//         setTimeout(
//             get_caption_mode, delay_time, object_caption, modeIndex, modeList
//         )
//     }
// }


// // document.getElementById("scene_caption").addEventListener("click", async function () {
// //     if (selected_scenes_list.value in scene_datas) {
// //         let scene_caps = scene_datas[selected_scenes_list.value]['scene_cap'];
// //         cleanChat();
// //         renderMessageEle(processMessage(scene_caps, "scene_cap"), "scene_cap");
// //     } else {
// //         cleanChat();
// //         renderMessageEle("Please select a scene first!", "default");
// //     }
// // });

// function checkAndExecuteSeqGrounding() {
//     if (selected_scenes_list.value in scene_datas) {
//         let tasks = scene_datas[selected_scenes_list.value]['seq_grounding'];
//         cleanChat();
//         renderMessageEleSeq(tasks, "seq_grounding");
//         // get_caption_mode(scene_datas[selected_scenes_list.value], 0, modeList);
//     } else {
//         cleanChat();
//         renderMessageEle("Please select a scene first!", "default");
//     }
// }

// function init_mesh_viewer(){
//     scene = new THREE.Scene();
//     scene.background = new THREE.Color(0xffffff);

//     // create a camera
//     camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
//     camera.lookAt(0, 0, 0);
//     camera.up.set(0, 0, 1);

//     tween = new TWEEN.Tween(camera.position);
//     // tween.to(endPos, view_time);
//     tween.easing(TWEEN.Easing.Quadratic.OutIn);

//     // add camera update tween 
//     init_tween();

//     renderer = new THREE.WebGLRenderer({
//         preserveDrawingBuffer: true
//     });
//     mesh_viewer = document.getElementById("mesh_viewer");
//     renderer.setSize(mesh_viewer.clientWidth, mesh_viewer.clientHeight);
//     console.log(renderer.domElement)
//     renderer.shadowMap.enabled = true;
//     renderer.shadowMapSoft = true;
//     document.getElementById("mesh_viewer").innerHTML = "";
//     document.getElementById("mesh_viewer").appendChild(renderer.domElement);

//     init_segmentation();
//     document.getElementById("segmentation").innerHTML = "";
//     document.getElementById("segmentation").appendChild(renderer_seg.domElement);

//     controls = new THREE.OrbitControls(camera, renderer.domElement);
//     controls.addEventListener('change', function(){
//         renderer.render(scene, camera);
//         renderer_seg.render(scene_seg, camera);
//     })
//     const animate = function () {
//         renderer.render(scene, camera);
//         renderer_seg.render(scene_seg, camera);
//         controls.update();
//         tween.update();
//         tweencamera.update();
//         set_basecam();
//         requestAnimationFrame(animate);
//     }
//     animate();
//     // add_label_listener();
// }
// function displayPointCloud(file, name, size=0.02, position=false, rotation=false, seg_flag = false) {
//     // scene.add(axesHelper);
//     // scene.add(gridHelper);
//     file = file.replace('/mnt/fillipo/scratch/masaccio/existing_datasets', './assets/datas');
//     var ext = file.split('.').pop();
//     file_path = file;
//     if(ext=='ply'){
//         loadPLYModel(file_path, function(mesh) {
//             mesh.name = name;
//             if(seg_flag){
//                 scene_seg.add(mesh);
//             }
//             else{
//                 scene.add(mesh);
//             }
//         });
//     }
//     else if(ext=='obj'){
//         loadOBJModel(file_path, function(mesh) {
//             mesh.name = name;
//             if(seg_flag){
//                 scene_seg.add(mesh);
//             }
//             else{
//                 scene.add(mesh);
//             }
//         });
//     }
// }
// function init_scene(){
//     console.log(selected_scenes_list.value)
//     var scene_name = selected_scenes_list.value;
//     console.log(scene_name)
//     var scene_info = scene_datas[scene_name];
//     cleanPointCloud();
//     cleanChat();
//     object_mesh_names = []
//     displayPointCloud(scene_info['mesh_file'], scene_name, size=0.02, position=scene_info['position'], rotation=scene_info['align_mat'], seg_flag=false);
//     displayPointCloud(scene_info['seg_file'], scene_name+'-seg', size=0.02, position=scene_info['position'], rotation=scene_info['align_mat'], seg_flag=true);
//     // for (var obj of scene_info['objects_info']){
//     //     var obj_name = obj['label'] + obj['id'];
//     //     displayPointCloud(obj['mesh'], obj_name, size=0.02, position=scene_info['position'], rotation=scene_info['align_mat']);
//     //     object_mesh_names.push(obj_name);
//     // }
// }

// async function load_datas(){
//     selected_scenes_list = document.getElementById("scene_list");
//     await loadLocalData();
//     for(var key in scene_datas){
//         var option = document.createElement("option");
//         option.text = key;
//         selected_scenes_list.add(option);
//     }
//     selected_scenes_list.selectedIndex = 1;
//     init_scene();
//     init_tween();
//     checkAndExecuteSeqGrounding();
// }

// document.getElementById("scene_list").addEventListener("change", async function () {
//     init_scene();
//     init_tween();
//     checkAndExecuteSeqGrounding();
// });

// window.addEventListener('resize',function() {
//     mesh_viewer = document.getElementById("mesh_viewer");
//     mesh_seg_viewer = document.getElementById("segmentation");

//     camera.aspect = mesh_viewer.clientWidth / mesh_viewer.clientHeight;
//     camera.updateProjectionMatrix();
//     renderer.setSize(mesh_viewer.clientWidth, mesh_viewer.clientHeight);
//     renderer_seg.setSize(mesh_seg_viewer.clientWidth, mesh_seg_viewer.clientHeight);
//     if(scene_graph) {
//         scene_graph.initalLayout.run();
//     }

// });

// function get_meshsize(mesh){
//     const box = new THREE.Box3().setFromObject(mesh);
//     const bsize = new THREE.Vector3();
//     box.getSize(bsize);
//     return bsize;
// }

// // function position_debug(position){
// //     // position test code 
// //     const geometry = new THREE.SphereGeometry(0.1, 32, 32);  
// //     const material = new THREE.MeshBasicMaterial({color: 0xff0000});
// //     const sphere = new THREE.Mesh(geometry, material);
// //     sphere.position.set(position.x, position.y, position.z);
// //     sphere.name = 'place';
// //     if(scene.getObjectByName(sphere.name)){
// //         scene.remove(scene.getObjectByName(sphere.name));
// //     }
// //     scene.add(sphere);
// // }

// function box_debug(position, size=(1, 1, 1), lineWidth=10, clear=true){
//     if(clear){
//         removeBboxesFromScene();
//     }
//     var pos = new THREE.Vector3(
//         position[0],
//         position[1],
//         position[2])
//     const geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
//     const edgesGeometry = new THREE.EdgesGeometry(geometry);
//     // const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: lineWidth });
//     const lineMaterial = new THREE.LineBasicMaterial({ color: 'red'});
//     lineMaterial.linewidth = lineWidth;
//     const boxLine = new THREE.LineSegments(edgesGeometry, lineMaterial);
//     boxLine.name = 'show_box';
//     boxLine.position.copy(pos);
    
//     // var path = new THREE.CurvePath();
//     // path.add(new THREE.LineCurve3(new THREE.Vector3(position[0], position[1], position[2]), new THREE.Vector3(size[0], size[1], size[2])));
//     // var geometry = new THREE.TubeGeometry(path, 20, lineWidth, 8, false);
//     // var material = new THREE.MeshBasicMaterial({ color: 'red' });
//     // var mesh = new THREE.Mesh(geometry, material);
//     // scene.add(mesh);

//     // var dir = new THREE.Vector3(0, 0, -1);
//     // dir.normalize();
//     // var origin = new THREE.Vector3(pos.x, pos.y, pos.z + size[2] / 2 + 1.0);
//     // var length = 1;
//     // var hex = 0xffff00;
//     // var arrowHelper = new THREE.ArrowHelper(dir, origin, length, hex);
//     // scene.add(arrowHelper);

//     return boxLine;
// }


// function removeBboxesFromScene() {
//     const bboxes = scene.children.filter(child => child.name === "show_box");
//     for (const bbox of bboxes) {
//         scene.remove(bbox);
//     }
// }

// function camera_look_target(size, position, duration){
//     max_size = Math.max(size[0], size[1], size[2]);
//     max_size *= 2.0;
//     // length = (max_size/2)/Math.tan(45/2)*2;
//     length = 5
//     const dir = new THREE.Vector3(
//             -Math.sign(position[0]),
//             -Math.sign(position[1]),
//             position[2]).normalize();
    
//     var tx = position[0] + dir.x*length;
//     var ty = position[1] + dir.y*length;
//     var tz = position[2] + size[2]/2 +2.0;
//     // var tz = position[2] + dir.z*length;

//     // // Assume obstacles is an array of meshes that might obstruct the view
//     // var obstacles = scene.children;
//     // findUnobstructedView(new THREE.Vector3(tx, ty, tz), obstacles);
//     // var cameraPosition = camera.position.clone();

//     tweencamera.stop();
//     tweencamera = new TWEEN.Tween({
//         x: camera.position.x, y: camera.position.y, z: camera.position.z,
//         lookAtX: controls.target.x, lookAtY: controls.target.y, lookAtZ: controls.target.z
//     });
//     tweencamera.to({x: tx, y: ty, z: tz, lookAtX:  position[0], lookAtY:  position[1], lookAtZ:  position[2]}, duration);
//     tweencamera.onUpdate(updateCamera);
// }

// const updateCamera = function (object, elapsed) {
//     camera.position.set(object.x, object.y, object.z);
//     controls.target = new THREE.Vector3(object.lookAtX, object.lookAtY, object.lookAtZ);
// };


// function findUnobstructedView(target, obstacles) {
//     var direction = new THREE.Vector3().subVectors(target, camera.position).normalize();
//     var raycaster = new THREE.Raycaster(camera.position, direction);
//     var intersects = raycaster.intersectObjects(obstacles);

//     while (intersects.length > 0) {
//         // Move the camera slightly towards the target
//         camera.position.add(direction.multiplyScalar(0.1));
//         // Update the raycaster and check for intersections again
//         raycaster.set(camera.position, direction);
//         intersects = raycaster.intersectObjects(obstacles);
//     }

//     // Now the camera should have an unobstructed view of the target
//     camera.lookAt(target);
// }

// function init_tween(){
//     var start_pos = [3, 3, 3]
//     var end_pos = [0, 0, 0]
//     tweencamera = new TWEEN.Tween({x: start_pos[0], y: start_pos[1], z: start_pos[2], lookAtX: end_pos[0], lookAtY: end_pos[1], lookAtZ: end_pos[2]});
//     camera.position.set(1, 1, -1);
//     tweencamera.to({x: start_pos[0], y: start_pos[1], z: start_pos[2], lookAtX: end_pos[0], lookAtY: end_pos[1], lookAtZ: end_pos[2]}, view_time);
//     tweencamera.onUpdate(updateCamera);
//     tweencamera.start();
// }

// function init_segmentation(){
//     renderer_seg = new THREE.WebGLRenderer({
//         preserveDrawingBuffer: true
//     });
//     viewer = document.getElementById("segmentation");
//     renderer_seg.setSize(viewer.clientWidth, viewer.clientHeight);
//     renderer_seg.shadowMap.enabled = true;
//     renderer_seg.shadowMapSoft = true;
//     scene_seg = new THREE.Scene();
//     scene_seg.background = new THREE.Color(0xffffff);
// }

// function get_selected_dom(){
//     var selected_scenes = document.getElementById("default_example");
//     var x = selected_scenes.selectedIndex;
//     var y = selected_scenes.options[x];
//     return y;
// }

// function cleanPointCloud() {
//     scene.children = []
//     scene_seg.children = []
// }

// function set_basecam(){
//     cam_x = camera.position.x;
//     cam_y = camera.position.y;
//     cam_z = camera.position.z;
// }
// function set_light(){
//     const amblight = new THREE.AmbientLight( 'gray', 1.5);
//     scene.add( amblight );
//     const pointLight = new THREE.PointLight( 'gray', 1.5, 10 );
//     const sphereSize = 1;
//     const pointLightHelper = new THREE.PointLightHelper( pointLight, sphereSize );
//     // scene.add( pointLightHelper );
//     pointLight.position.set( 0, 0, 4 );
//     scene.add( pointLight );
// }

// function loadOBJModel_new(fileURL, callback) {
//     THREE.DefaultLoadingManager.onLoad = function ( ) {
//         console.log( 'Loading completed');
//     };

//     THREE.DefaultLoadingManager.onProgress = function ( url, itemsLoaded, itemsTotal ) {
//         console.log( 'Loading file: ' + url + '.\nLoaded ' + itemsLoaded + ' of ' + itemsTotal + ' files.' );
//     };

//     THREE.DefaultLoadingManager.onError = function ( url ) {
//         console.error( 'There was an error loading ' + url );
//     };

//     fetch(fileURL)
//         .then(response => response.text())
//         .then(data => {
//             const mtlFileName = data.match(/mtllib (.+)/)[1];
//             const mtlFilePath = fileURL.replace(/[^/]*$/, mtlFileName);

//             let mtlLoader = new THREE.MTLLoader();
//             mtlLoader.load(mtlFilePath, function(materials) {
//                 materials.preload();

//                 for (let materialName in materials.materials) {
//                     let material = materials.materials[materialName];
//                     console.log('Material:', materialName);
//                     console.log('  Ambient map:', material.map);
//                     console.log('  Diffuse map:', material.map);

//                     if (material.map && material.map.image === undefined) {
//                         console.error('The image data of the texture has not been loaded:', material.map);
//                     }
//                 }

//                 let loader = new THREE.OBJLoader();
//                 loader.setMaterials(materials);
//                 loader.load(
//                     fileURL,
//                     function (geometry) {
//                         geometry.scale.multiplyScalar(1);      
//                         target_size = get_meshsize(geometry);
//                         if(position){
//                             geometry.position.x = position[0];
//                             geometry.position.y = position[1];
//                             geometry.position.z = position[2];
//                         };
//                         if(rotation){
//                             // geometry.quaternion.set(rotation[0], rotation[1], rotation[2], rotation[3])
//                             radians = THREE.MathUtils.degToRad(rotation)
//                             geometry.rotation.z = -radians
//                         }
//                         callback(geometry);
//                     },
//                     function (xhr) {
//                         console.log((xhr.loaded / xhr.total * 100) + '% loaded');
//                     },
//                     function (error) {
//                         console.error('There was an error loading the model:', error);
//                     }
//                 );
//             },
//             function (xhr) {
//                 console.log('MTL file ' + (xhr.loaded / xhr.total * 100) + '% loaded');
//                 console.log('Response URL:', xhr.target.responseURL);
//             }, 
//             function (error) {
//                 console.error('There was an error loading the MTL file:', error);
//                 console.error('Error details:', error.detail);
//                 if (error.detail instanceof Event && error.detail.target instanceof Image) {
//                     console.error('Failed to load image:', error.detail.target.src);
//                 }
//             });
//         });
// }

// function loadOBJModel(fileURL, callback) {
//     let loader = new THREE.OBJLoader();
//     loader.load(
//         fileURL,
//         function (geometry) {
//             const material = new THREE.PointsMaterial({
//                 size: size,
//             });
//             material.vertexColors = true;
//             geometry.scale.multiplyScalar(1);      
//             target_size = get_meshsize(geometry);
//             if(position){
//                 geometry.position.x = position[0];
//                 geometry.position.y = position[1];
//                 geometry.position.z = position[2];
//             };
//             if(rotation){
//                 // geometry.quaternion.set(rotation[0], rotation[1], rotation[2], rotation[3])
//                 radians = THREE.MathUtils.degToRad(rotation)
//                 geometry.rotation.z = -radians
//             }
//             geometry.material = material;
//             geometry.traverse( child => {
//                 if (child.material) {
//                     child.material = material;
//                   }
//             } );
//             callback(geometry);
//         },
//         function (xhr) {
//             console.log((xhr.loaded / xhr.total * 100) + '% loaded');
//         },
//         function (error) {
//             console.error('There was an error loading the model:', error);
//         }
//     );
// }

// function loadPLYModel(fileURL, callback) {
//     let loader = new THREE.PLYLoader();
//     loader.load( fileURL, 
//         function ( mesh ) {
//             var material = new THREE.PointsMaterial( {size: size} );
//             material.vertexColors = true;
//             var mesh = new THREE.Mesh( mesh, material);
//             target_size = get_meshsize(mesh);
//             mesh.scale.multiplyScalar(1);
//             mesh.castShadow = true;
//             mesh.receiveShadow = true;
//             if(position){
//                 mesh.position.x = position[0];
//                 mesh.position.y = position[1];
//                 mesh.position.z = position[2];
//             };
//             if(rotation){
//                 radians = THREE.MathUtils.degToRad(rotation)
//                 mesh.rotation.z = -radians
//                 // mesh.quaternion.set(rotation[0], rotation[1], rotation[2], rotation[3])
//             }
//             callback(mesh);
//         },
//         function (xhr) {
//             console.log((xhr.loaded / xhr.total * 100) + '% loaded');
//         },
//         function (error) {
//             console.error('There was an error loading the model:', error);
//         }
//     );
// }

// function camera_breathe(){
//     if(breathe){
//         mean_size = (target_size.x + target_size.y + target_size.z)/3;

//         b = Math.sin(delta)*mean_size/3000
//         const dir = new THREE.Vector3(
//             camera.position.x - controls.target.x, 
//             camera.position.y - controls.target.y,
//             camera.position.z - controls.target.z).normalize();
//         camera.position.x = cam_x + b * dir.x
//         camera.position.y = cam_y + b * dir.y 
//         camera.position.z = cam_z + b * dir.z
//         delta += 0.1
//     }
// }

// // function add_label_listener(){
// //     renderer.domElement.addEventListener('dblclick', onMouseDown, false);
// //     function onMouseDown(event) {
// //         // if (event.button !== 1) return;
// //         var mesh_viewer = document.getElementById('mesh_viewer');
// //         var rect = mesh_viewer.getBoundingClientRect();
// //         // var selectedArrow;
// //         const mouse = new THREE.Vector2();
// //         const raycaster = new THREE.Raycaster();
// //         mouse.x = ((event.clientX - rect.left ) / rect.width )*2 - 1;
// //         mouse.y = -((event.clientY - rect.top ) / rect.height)*2 + 1;
// //         console.info(mouse.x, mouse.y);
// //         raycaster.setFromCamera(mouse, camera);

// //         // debug ray 
// //         var dir = raycaster.ray.direction.clone().multiplyScalar(10); // length of 1000 units
// //         var arrow = new THREE.ArrowHelper(raycaster.ray.direction, raycaster.ray.origin, dir.length(), 0xff0000);
// //         var minSize = Number.MAX_SAFE_INTEGER;
// //         var minBbox = null;
// //         var minObjectInfo = null;
// //         for (var [index, object_name] of object_mesh_names.entries()){
// //             var meshes = scene.getObjectByName(object_name)
// //             var intersects = raycaster.intersectObjects(meshes.children)
// //             if (intersects.length == 0) {
// //                 meshes = scene.getChildByName(object_name);
// //                 intersects = raycaster.intersectObject(meshes);
// //             }
// //             if (intersects.length > 0){
// //                 console.log(object_name, intersects.length);
// //                 const position = intersects[0].point;
// //                 position_debug(position);
// //                 var object_info = scene_datas[selected_scenes_list.value]['objects_info'][index];
// //                 camera_look_target(object_info['size'], object_info['position'], 1000);
// //                 var boundingBox = new THREE.Box3().setFromObject(meshes);

// //                 var center = new THREE.Vector3();
// //                 var bboxSize = new THREE.Vector3();
// //                 boundingBox.getCenter(center);
// //                 boundingBox.getSize(bboxSize);
                
// //                 console.log("bbox size", bboxSize["x"]);
// //                 bboxSizeScalar = bboxSize["x"] * bboxSize["y"] * bboxSize["z"];
// //                 console.log(bboxSizeScalar);
// //                 if (bboxSizeScalar < minSize) {
// //                     minBbox = box_debug(object_info["position"], object_info["size"], 5);
// //                     minObjectInfo = object_info;
// //                 }
// //             }
// //         }
// //         if (minBbox != null) {
// //             scene.add(minBbox);
// //             tweencamera.start();
// //             cleanChat();
// //             console.log("calling caption:")
// //             console.log(minBbox)
// //             console.log(minObjectInfo)
// //             get_caption_mode(minObjectInfo['caption'], 0, modeList);
// //         }
// //     }
// // }

// function view_scene(){
//     if(breathe){
//         camera.position.set(distance*Math.sin(rotation_r), distance*Math.cos(rotation_r), 4.5);
//         controls.target.set(0, 0, 0.5);
//         rotation_r += 0.01;
//     }
// }

// function main(){
//     init_mesh_viewer();
//     load_datas();
// }
// main()
