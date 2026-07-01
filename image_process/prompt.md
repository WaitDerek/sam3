# 物料分割 Prompt 参考

本文件收集 15 种物料和 5 种料箱的 SAM3 文本提示词。实际运行时可按场景组合 2-4 条 prompt，并配合面积、数量和亮色排除参数使用。

通用排除词：

```text
object only, not crate, not tray wall, not foam, not table, not reflection
separate each visible object, do not merge touching objects
if heavily occluded and object boundary is unclear, return no target
```

## 黑色小件与塑料件

### 1. 空滤器进气连接管总成

```text
black S-curved automotive air intake hose with corrugated middle
black ribbed flexible automotive intake duct
black plastic intake duct with flared mouth, object only, not crate
single black plastic air intake elbow with corrugated hose
```

### 2. 空调新风口总成

```text
black rectangular hollow automotive HVAC fresh air inlet plastic part
black plastic fresh air inlet housing, fill the hollow center as part of the object
black rectangular duct frame with foam wrapped edges, object only, not white foam pad
black HVAC fresh air inlet in divided crate, segment the whole part, not crate grid
```

### 3. 背门驱动机构撑杆上球头销支架（左）

```text
small black ball stud bracket in blue divided foam tray
black metal or plastic bracket part inside blue foam grid compartment
one or two small black bracket objects only, not blue tray wall
black target object on blue foam background, exclude upper-left unrelated dark area
```

### 4. 前保险杠侧安装支架总成

```text
small black plastic bumper mounting bracket
black automotive side bumper bracket on green table
small dark plastic bracket centered on teal background
black bumper bracket inside crate, object only, not crate divider
```

### 5. 螺钉盖板

```text
small black screw cover plastic plate
black rectangular screw cover cap, object only
small black plastic cover plate on table
black screw cover in crate compartment, not crate wall
```

### 6. 泡沫块

```text
white or light foam block object only
rectangular foam packing block with clear boundary
single foam block on table, not background
foam insert piece inside crate, not crate wall
```

### 7. A立柱

```text
automotive A-pillar trim part
long narrow interior pillar plastic trim
dark or light A-pillar cover component, object only
A-pillar trim in crate, not crate divider or foam pad
```

### 8. 主雨刮臂总成

```text
long black automotive main windshield wiper arm assembly
black metal wiper arm with pivot hub and hook connector end, object only
rigid black windshield wiper arm assembly, not rubber wiper blade alone
main wiper arm on table, include the arm body and base joint, not shadow
main wiper arm inside crate, separate each visible arm, not crate rack or divider
```

### 9. 车标

```text
silver automotive logo emblem
chrome car badge emblem, object only, not reflection
two visible silver automotive logo emblems, separate each object
if two emblems touch and cannot be separated cleanly, send to manual review
```

### 10. 低音电喇叭总成

```text
black round automotive low-tone horn assembly
circular black electric horn with metal bracket
black disc-shaped car horn, object only
two black horn assemblies in crate, separate each visible object
```

### 11. 副电动车窗开关总成

```text
black automotive passenger power window switch assembly
small black rectangular window switch module
black plastic switch component with button surface
window switch in tray, object only, not tray wall
```

### 12. 背门自动开闭系统ECU控制器总成

```text
black rectangular automotive ECU controller module
power tailgate ECU controller box with connector area
black electronic control unit housing, object only
ECU controller in crate compartment, not crate divider
```

### 13. 副雨刮器

```text
black automotive windshield wiper blade
long black rubber wiper blade assembly
curved black car wiper blade with center connector
two black windshield wiper blades, separate each blade, not crate rack
```

### 14. 洗涤器水壶加注管总成

```text
black automotive washer fluid filler neck pipe
long black plastic washer filler tube
curved black filler pipe with cap or mounting end
washer filler pipe in crate, object only, not crate divider
```

### 15. 三角警告牌

```text
red triangular warning triangle object
folded automotive warning triangle frame
red reflective warning triangle, object only
warning triangle in crate or on table, not background
```

## 料箱

料箱 prompt 用于识别整个料箱本体。若目标被遮挡超过约 40%，不识别被严重遮挡的料箱；其余可见料箱应分别识别。

### 16. 料箱1

```text
gray plastic storage crate box1, whole crate only
visible plastic crate with grid compartments, include outer rim and dividers
two visible box1 crates, separate each crate if not heavily occluded
```

### 17. 料箱2

```text
gray plastic storage crate box2, whole crate only
plastic divided material box, include all visible crate walls
two visible box2 crates, ignore crate mostly hidden behind another crate
```

### 18. 料箱3

```text
blue or gray divided foam material box3, whole container only
storage crate with multiple compartments, include outer boundary
box3 crate only, not objects inside the compartments
```

### 19. 料箱4

```text
plastic material crate box4, whole crate only
large storage box with grid cells and visible outer rim
box4 crate, ignore contents and foam inserts
```

### 20. 料箱5

```text
plastic material crate box5, whole crate only
visible divided storage container, include complete visible crate
two visible box5 crates, separate each unoccluded crate
```

## 料箱中物料的通用补充

这些 prompt 用于识别料箱里的物料，不用于识别料箱本体：

```text
black target object inside blue foam grid compartment, not blue tray wall
black target object on white foam pad, not white foam
one or two visible target objects in divided crate, separate each object
target object only, exclude crate wall, divider, foam pad and shadow
```
